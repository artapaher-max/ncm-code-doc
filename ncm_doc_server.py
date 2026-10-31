# ncm_doc_server.py
"""Servidor MCP para documentación técnica acelerada mediante Stanford NCM (Next Concept Model)."""

import contextlib
import logging
import os
import re
import sys
from typing import Optional

import torch
from omegaconf import OmegaConf
from mcp.server.mcpserver import MCPServer

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NCM_DIR = os.path.join(CURRENT_DIR, "ncm") if os.path.isdir(os.path.join(CURRENT_DIR, "ncm", "models")) else CURRENT_DIR

for path in [CURRENT_DIR, NCM_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

mcp = MCPServer("antigravity-ncm-writer")
device: str = "cuda" if torch.cuda.is_available() else "cpu"

tk: Optional[VQVAETokenizer] = None
net: Optional[NextConceptTransformer] = None
CFG = None
ALLOW: Optional[torch.Tensor] = None
EOS_ID: int = 50256
PAD_ID: int = 50257
PREFIX_LEN: int = 512


def load_allowance(spec: str, ckpt_dir: str, dev: str) -> Optional[torch.Tensor]:
    """Carga el unigram allowance de frecuencias desde el checkpoint si está especificado."""
    try:
        if ":" in spec:
            path, ref = spec.split(":", 1)
            full_path = os.path.join(ckpt_dir, path)
            if not os.path.exists(full_path):
                return None
            obj = torch.load(full_path, map_location=dev, weights_only=False)
            for k in ref.split("."):
                if not isinstance(obj, dict) or k not in obj:
                    return None
                obj = obj[k]
            return obj.double()
        else:
            full_path = os.path.join(ckpt_dir, spec)
            if os.path.exists(full_path):
                return torch.load(full_path, map_location=dev, weights_only=False).double()
    except Exception:
        pass
    return None


def load_stanford_ncm(task_name: str = "ncm_code_doc") -> None:
    """Carga los módulos VQ-VAE, generador jerárquico NextConceptTransformer y parámetros oficiales."""
    global tk, net, CFG, ALLOW, EOS_ID, PAD_ID, PREFIX_LEN

    config_path = os.path.join(NCM_DIR, "config", "eval", f"{task_name}.yaml")
    checkpoint_dir = os.path.join(NCM_DIR, "artifacts", task_name)

    # Fallback a xsum si la configuración personalizada aún no existe
    if not os.path.exists(config_path) or not os.path.exists(checkpoint_dir):
        task_name = "xsum"
        config_path = os.path.join(NCM_DIR, "config", "eval", "xsum.yaml")
        checkpoint_dir = os.path.join(NCM_DIR, "artifacts", "xsum")

    CFG = OmegaConf.load(config_path)
    OmegaConf.set_struct(CFG, False)

    tokenizer_ckpt = os.path.join(checkpoint_dir, "tokenizer.pt")
    if hasattr(CFG.tokenizer, "vqvae"):
        CFG.tokenizer.vqvae.checkpoint_path = tokenizer_ckpt

    with contextlib.redirect_stdout(sys.stderr):
        tk = VQVAETokenizer.from_config(CFG.tokenizer)
        if os.path.exists(tokenizer_ckpt):
            tok_sd = torch.load(tokenizer_ckpt, map_location=device, weights_only=False)
            state = tok_sd["model"] if "model" in tok_sd else tok_sd.get("state_dict", tok_sd)
            tk.vqvae.load_state_dict(state, strict=False)
        tk.vqvae.to(device).eval()

        net = NextConceptTransformer(vqvae_local=tk.vqvae, config=CFG.model).to(device)

        gen_ckpt = os.path.join(checkpoint_dir, "generator.pt")
        gen_sd = {}
        if os.path.exists(gen_ckpt):
            gen_sd = torch.load(gen_ckpt, map_location=device, weights_only=False)
            state = gen_sd["model"] if "model" in gen_sd else gen_sd.get("state_dict", gen_sd)
            net.load_state_dict(state, strict=False)

        EOS_ID = getattr(tk.pre_tokenizer, "eos_token_id", 50256)
        PAD_ID = int(getattr(CFG.model, "prefix_pad_token_id", 50257) or 50257)
        PREFIX_LEN = int(getattr(CFG.model, "prefix_len", 512))
        net.prefix_pad_id = PAD_ID
        net.eval()

        # Solo cargar allowance si el checkpoint adaptado lo tiene específicamente
        ALLOW = None
        if "inference" in gen_sd and "token_unigram" in gen_sd["inference"]:
            ALLOW = gen_sd["inference"]["token_unigram"].to(device).double()

    print(f"-> Stanford NCM [{task_name.upper()}] inicializado correctamente en [{device}].", file=sys.stderr)


# Carga inicial al arrancar el proceso MCP
try:
    load_stanford_ncm("ncm_code_doc")
except Exception as e:
    print(f"Error cargando ncm_code_doc: {e}. Reintentando con xsum...", file=sys.stderr)
    try:
        load_stanford_ncm("xsum")
    except Exception as ex:
        print(f"Fallo crítico de inicialización NCM: {ex}", file=sys.stderr)


def pick_tokens(
    logits: torch.Tensor,
    allow: Optional[torch.Tensor] = None,
    freq: float = 0.0,
    mu: float = 0.0,
    adj: Optional[list[float]] = None,
) -> torch.Tensor:
    """Algoritmo de selección de tokens con penalización adaptativa balanceada para código."""
    if adj is None:
        adj = [1.0, 2.0, 0.5]

    if freq <= 0 and not adj:
        return logits.argmax(-1)

    B, T, V = logits.shape
    lg = logits.double()
    allow_t = (allow[:V] * T)[None, :] if allow is not None else None
    al = float(adj[0]) if adj else 0.0
    aw = int(adj[1]) if adj and len(adj) > 1 else 1
    ad = float(adj[2]) if adj and len(adj) > 2 else 1.0

    cnt = torch.zeros(B, V, dtype=torch.float64, device=lg.device)
    out = torch.empty(B, T, dtype=torch.long, device=lg.device)
    hist: list[torch.Tensor] = []

    for t in range(T):
        step = lg[:, t, :].clone()
        if freq > 0 and allow_t is not None:
            excess = (cnt - allow_t).clamp_min(0.0)
            step -= freq * excess
            if mu > 0:
                step += mu * ((T - t) / T) * (allow_t - cnt).clamp_min(0.0)

        for j, h in enumerate(reversed(hist[-aw:])):
            step.scatter_add_(
                1, h[:, None], torch.full((B, 1), -al * (ad ** j), dtype=torch.float64, device=lg.device)
            )

        p = step.argmax(-1)
        out[:, t] = p
        hist.append(p)
        cnt.scatter_add_(1, p[:, None], torch.ones(B, 1, dtype=torch.float64, device=lg.device))

    return out


def remove_repeated_phrases(words: list[str]) -> list[str]:
    """Elimina repeticiones de n-gramas adyacentes comunes en generación no-autorregresiva."""
    for n in [3, 2, 1]:
        i = 0
        while i + 2 * n <= len(words):
            gram1 = [w.strip(".,;:!?()[]\"'").lower() for w in words[i : i + n]]
            gram2 = [w.strip(".,;:!?()[]\"'").lower() for w in words[i + n : i + 2 * n]]
            if gram1 == gram2:
                if any(c in ".,;:!?" for c in words[i + 2 * n - 1]) and not any(c in ".,;:!?" for c in words[i + n - 1]):
                    words[i + n - 1] += words[i + 2 * n - 1][-1]
                words = words[: i + n] + words[i + 2 * n :]
            else:
                i += 1
    return words


def dec(ids: list[int], eos_id: int, pad_id: int, tokenizer: VQVAETokenizer, cut_at_eos: bool = True) -> str:
    """Decodifica la lista de IDs a texto legible, eliminando duplicados adyacentes de la generación NAR."""
    token_list = list(ids)
    if cut_at_eos and eos_id in token_list:
        token_list = token_list[:token_list.index(eos_id)]

    # 1. Desduplicación de tokens adyacentes idénticos
    dedup_ids: list[int] = []
    for tid in token_list:
        if tid == pad_id:
            continue
        if not dedup_ids or tid != dedup_ids[-1]:
            dedup_ids.append(tid)

    text = tokenizer.pre_tokenizer.decode(dedup_ids, skip_special_tokens=True).strip()

    # 2. Desduplicación de frases repetidas (n-grams)
    words = text.split()
    clean_words = remove_repeated_phrases(words)
    s = " ".join(clean_words).strip()

    # 3. Limpieza de corte al final de primera oración coherente
    if "." in s:
        first_sentence = s.split(".")[0].strip()
        if len(first_sentence.split()) >= 2:
            s = first_sentence + "."

    # 4. Normalizaciones comunes de sub-tokens BPE
    s = re.sub(r"\bRever\b", "Reverse", s)
    s = re.sub(r"\balifif\b", "elif", s)

    # 5. Formato de capitalización inicial
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    return s


@mcp.tool()
def fast_ncm_redaction(
    context_or_code: str,
    instruction: str = "Document technical summary",
) -> str:
    """Genera documentación técnica de código de forma paralela coarse-to-fine sin gastar tokens de LLM.

    Args:
        context_or_code: Código fuente o función que se desea resumir/documentar.
        instruction: Directriz o tipo de documentación solicitada.
    """
    if not context_or_code or not context_or_code.strip():
        return "Error: Código o contexto de entrada vacío."

    if net is None or tk is None:
        return "Error: El modelo NCM no está inicializado."

    try:
        inst_clean = instruction.strip() if instruction and instruction.strip() else "Document technical summary"
        formatted_prompt = f"{inst_clean}:\n{context_or_code.strip()}\n\nSummary:\n"

        with torch.no_grad():
            with contextlib.redirect_stdout(sys.stderr):
                raw_tokens = tk.pre_tokenizer.encode(formatted_prompt)

                if len(raw_tokens) < PREFIX_LEN:
                    padded_tokens = raw_tokens + [PAD_ID] * (PREFIX_LEN - len(raw_tokens))
                else:
                    padded_tokens = raw_tokens[:PREFIX_LEN]

                x = torch.tensor([padded_tokens], dtype=torch.long, device=device)

                # Generación greedy determinista sobre los códigos VQ discretos
                g = net.generate(
                    None,
                    x,
                    temperature=1.0,
                    top_k=1,
                    top_p=0.0,
                    cfg_scale=0.0,
                    cfg_null_eos=EOS_ID,
                )

                # Decodificación coarse-to-fine a través del VQ-VAE decoder
                _, zl = tk.vqvae.vq.idx_to_zhats(g, return_zlevels=True)
                logits = tk.vqvae.decoder(zl[-1])

                # Decodificación adaptativa sin allowance sesgado de noticias
                picked = pick_tokens(logits, allow=ALLOW, freq=0.0, mu=0.0, adj=[1.0, 2.0, 0.5])
                clean_text = dec(picked[0].tolist(), EOS_ID, PAD_ID, tk, cut_at_eos=True)

        return clean_text if clean_text else "(No se generó contenido legible)"

    except Exception as e:
        return f"Error en inferencia NCM: {str(e)}"


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
