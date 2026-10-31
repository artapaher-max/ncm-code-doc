# finetune_ncm.py
"""
Script de Fine-Tuning para Stanford NCM (Next Concept Model)
Especializado en documentación técnica de código fuente.
"""

import contextlib
import json
import os
import shutil
import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NCM_DIR = os.path.join(CURRENT_DIR, "ncm") if os.path.isdir(os.path.join(CURRENT_DIR, "ncm", "models")) else CURRENT_DIR

for path in [CURRENT_DIR, NCM_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from models.next_concept import NextConceptTransformer
from tokenization.vqvae_tokenizer import VQVAETokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 5
BATCH_SIZE = 2           # Óptimo para GPUs de 6 GB VRAM (~2.9 GB utilizados)
GRAD_ACCUM_STEPS = 2     # Tamaño de lote efectivo = 4
LR = 5e-5
PREFIX_LEN = 512         # Contexto de entrada (código)
TARGET_LEN = 48          # Contexto del VQ-VAE decoder (resumen generado)
PAD_ID = 50257           # prefix_pad_token_id en config
EOS_ID = 50256           # <|endoftext|> en GPT-2


class CodeDocDataset(Dataset):
    """Dataset que separa estrictamente el prefijo de entrada (código) del objetivo (docstring)."""

    def __init__(self, jsonl_path: str, tokenizer: VQVAETokenizer):
        self.samples = []
        self.tk = tokenizer

        print(f"-> Cargando dataset desde {jsonl_path}...", file=sys.stderr)
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"No se encontró el dataset en: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                code_text = item.get("prompt", "").strip()
                doc_text = item.get("target", "").strip()

                if not code_text or not doc_text:
                    continue

                # 1. Prefijo: Prompt estructurado idéntico al servidor de inferencia
                prefix_text = f"Document technical summary:\n{code_text}\n\nSummary:\n"
                p_tokens = self.tk.pre_tokenizer.encode(prefix_text)
                if len(p_tokens) < PREFIX_LEN:
                    p_mask = [False] * len(p_tokens) + [True] * (PREFIX_LEN - len(p_tokens))
                    p_tokens = p_tokens + [PAD_ID] * (PREFIX_LEN - len(p_tokens))
                else:
                    p_tokens = p_tokens[:PREFIX_LEN]
                    p_mask = [False] * PREFIX_LEN

                # 2. Objetivo: Documentación técnica (<= 48 tokens con EOS)
                t_tokens = self.tk.pre_tokenizer.encode(doc_text)
                if len(t_tokens) < TARGET_LEN:
                    t_tokens = t_tokens + [EOS_ID] + [PAD_ID] * (TARGET_LEN - len(t_tokens) - 1)
                else:
                    t_tokens = t_tokens[:TARGET_LEN - 1] + [EOS_ID]

                self.samples.append({
                    "prefix_ids": torch.tensor(p_tokens, dtype=torch.long),
                    "prefix_pad_mask": torch.tensor(p_mask, dtype=torch.bool),
                    "target_ids": torch.tensor(t_tokens, dtype=torch.long)
                })

        print(f"-> Total de muestras válidas: {len(self.samples)}", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]


def main() -> None:
    print(f"-> Inicializando Fine-Tuning de Stanford NCM en dispositivo: {DEVICE}", file=sys.stderr)
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # Resolver rutas
    config_path = os.path.join(NCM_DIR, "config", "eval", "xsum.yaml")
    checkpoint_dir = os.path.join(NCM_DIR, "artifacts", "xsum")

    cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(cfg, False)

    tokenizer_ckpt = os.path.join(checkpoint_dir, "tokenizer.pt")
    if hasattr(cfg.tokenizer, "vqvae"):
        cfg.tokenizer.vqvae.checkpoint_path = tokenizer_ckpt

    # 1. Cargar Tokenizer VQ-VAE
    print("-> Inicializando Tokenizer VQ-VAE...", file=sys.stderr)
    tk = VQVAETokenizer.from_config(cfg.tokenizer)
    tk.vqvae.to(DEVICE)

    # Congelar Encoder y Codebook (preserva espacio latente), descongelar Decoder (adapta vocabulario a código)
    for p in tk.vqvae.encoder.parameters():
        p.requires_grad = False
    for p in tk.vqvae.vq.parameters():
        p.requires_grad = False
    for p in tk.vqvae.decoder.parameters():
        p.requires_grad = True

    tk.vqvae.encoder.eval()
    tk.vqvae.vq.eval()
    tk.vqvae.decoder.train()

    # 2. Cargar Transformer Generador
    print("-> Inicializando NextConceptTransformer...", file=sys.stderr)
    net = NextConceptTransformer(vqvae_local=tk.vqvae, config=cfg.model).to(DEVICE)
    gen_ckpt = os.path.join(checkpoint_dir, "generator.pt")
    gen_sd = {}
    if os.path.exists(gen_ckpt):
        gen_sd = torch.load(gen_ckpt, map_location=DEVICE, weights_only=False)
        state = gen_sd["model"] if "model" in gen_sd else gen_sd.get("state_dict", gen_sd)
        net.load_state_dict(state, strict=False)
    net.train()

    # 3. Cargar Dataset
    dataset_path = os.path.join(NCM_DIR, "data_train.jsonl")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(CURRENT_DIR, "data_train.jsonl")

    dataset = CodeDocDataset(dataset_path, tk)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 4. Optimizador y Criterios
    trainable_params = list(net.parameters()) + list(tk.vqvae.decoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda") if DEVICE == "cuda" else None

    criterion_ncm = nn.CrossEntropyLoss()
    criterion_dec = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    print(f"-> Entrenando {EPOCHS} épocas (Batch size: {BATCH_SIZE}, Grad accum: {GRAD_ACCUM_STEPS})...", file=sys.stderr)
    for epoch in range(EPOCHS):
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(loader):
            prefix_ids = batch["prefix_ids"].to(DEVICE)
            prefix_pad_mask = batch["prefix_pad_mask"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            B = prefix_ids.shape[0]

            with torch.no_grad():
                # Prefijo crudo procesado por la capa wte del encoder VQ-VAE
                z_prefix = tk.vqvae.encoder.wte(prefix_ids)  # [B, 512, 384]

                # Códigos objetivo discretos en los 9 niveles VQ [B, 200]
                codes = tk.vqvae.encode(target_ids)

                # Entradas de los niveles 1..8 proyectadas [B, 196, 384]
                ncm_in = tk.vqvae.vq.idx_to_ncm_input(codes)

                # Latente del nivel final para entrenar el decoder [B, 48, 384]
                _, zlevels = tk.vqvae.vq.idx_to_zhats(codes, return_zlevels=True)
                z_final = zlevels[-1]

            autocast_ctx = torch.amp.autocast("cuda", dtype=torch.float16) if DEVICE == "cuda" else contextlib.nullcontext()
            with autocast_ctx:
                # Query aprendida para el nivel 0 [B, 4, 384]
                q0 = net.build_l0_query_input(B, z_prefix.dtype)

                # Secuencia de target completa: 4 + 196 = 200
                x_target = torch.cat([q0, ncm_in], dim=1)

                # Tensor total de entrada al Transformer: 512 + 200 = 712
                x = torch.cat([z_prefix, x_target], dim=1)

                # Forward del Transformer
                logits = net(x, prefix_pad_mask=prefix_pad_mask)
                loss_ncm = criterion_ncm(logits.view(-1, logits.size(-1)), codes.view(-1))

                # Adaptación de vocabulario en el decoder VQ-VAE
                dec_logits = tk.vqvae.decoder(z_final)
                loss_dec = criterion_dec(dec_logits.view(-1, dec_logits.size(-1)), target_ids.view(-1))

                loss = (loss_ncm + 0.5 * loss_dec) / GRAD_ACCUM_STEPS

            if scaler is not None:
                scaler.scale(loss).backward()
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss.backward()
                if (step + 1) % GRAD_ACCUM_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            total_loss += loss.item() * GRAD_ACCUM_STEPS

            if (step + 1) % 500 == 0:
                print(
                    f"   Época [{epoch+1}/{EPOCHS}] | Paso [{step+1}/{len(loader)}] "
                    f"| Loss NCM: {loss_ncm.item():.4f} | Loss Dec: {loss_dec.item():.4f}",
                    file=sys.stderr
                )

        avg_loss = total_loss / len(loader)
        print(f"==> Época [{epoch+1}/{EPOCHS}] Finalizada | Loss Promedio: {avg_loss:.4f}", file=sys.stderr)

    # 5. Guardar artefactos adaptados
    output_dir = os.path.join(NCM_DIR, "artifacts", "ncm_code_doc")
    os.makedirs(output_dir, exist_ok=True)

    # Preservar inference (token_unigram) para pick_tokens
    save_gen = {"model": net.state_dict()}
    if "inference" in gen_sd:
        save_gen["inference"] = gen_sd["inference"]

    torch.save(save_gen, os.path.join(output_dir, "generator.pt"))
    torch.save({"model": tk.vqvae.state_dict()}, os.path.join(output_dir, "tokenizer.pt"))

    # Crear configuración ncm_code_doc.yaml optimizada
    target_cfg = OmegaConf.load(config_path)
    OmegaConf.set_struct(target_cfg, False)
    target_cfg.task = "ncm_code_doc"
    target_cfg.checkpoints.generator = "generator.pt"
    target_cfg.checkpoints.tokenizer = "tokenizer.pt"
    if "generation" not in target_cfg or not target_cfg.generation:
        target_cfg.generation = OmegaConf.create({
            "temperature": 1.0,
            "top_k": 1,
            "top_p": 0.0,
            "cfg_scale": 0.0,
            "frequency_penalty": 2.0,
            "frequency_deficit": 1.0,
            "adjacency_penalty": "5,4,0.5",
            "frequency_allowance": "generator.pt:inference.token_unigram",
            "cut_at_eos": True,
        })
    else:
        target_cfg.generation.cut_at_eos = True

    cfg_out_dir = os.path.join(NCM_DIR, "config", "eval")
    os.makedirs(cfg_out_dir, exist_ok=True)
    OmegaConf.save(target_cfg, os.path.join(cfg_out_dir, "ncm_code_doc.yaml"))

    print(f"-> ¡Fine-Tuning completado exitosamente! Pesos y config guardados en: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
