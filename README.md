# NCM Code Documentation Server

Servidor MCP (Model Context Protocol) local para generación rápida de documentación y docstrings técnicos en código fuente, basado en la arquitectura **Next-Concept Modeling (NCM)** desarrollada por Stanford Hazy Research.

El modelo opera de manera **no autorregresiva (NAR)** mediante un espacio latente jerárquico (VQ-VAE de 9 niveles). A diferencia de los LLMs tradicionales autorregresivos que generan texto token a token, NCM decodifica la representación conceptual en paralelo, permitiendo generar resúmenes técnicos en ~50–100 ms sobre GPU local y reduciendo el consumo de tokens en llamadas a modelos frontera (Claude, Gemini, GPT-4).

## Características

- **Inferencia no autorregresiva**: Decodificación en paralelo a través de 9 niveles jerárquicos de cuantización vectorial.
- **Rendimiento local**: Diseñado para ejecutarse en GPUs de consumo (e.g. NVIDIA RTX 2060 de 6 GB VRAM) con baja sobrecarga de memoria (~2.5 GB en inferencia).
- **Alcance de salida**: Hasta 48 tokens por pasada (límite del decodificador jerárquico VQ-VAE). Especializado en docstrings de una línea, firmas y resúmenes funcionales concisos.
- **Protocolo MCP estándar**: Conexión vía stdio compatible con Antigravity IDE, Claude Desktop y clientes compatibles con el SDK de MCP.
- **Posprocesamiento de código**: Limpieza de n-gramas repetidos generados por decodificación NAR y reconstrucción de subtokens BPE.

## Instalación

### Requisitos previos
- Python 3.10 o superior
- PyTorch con soporte CUDA (GPU con mínimo 4–6 GB VRAM recomendada)
- Git

### 1. Clonar el repositorio
```bash
git clone https://github.com/artapaher-max/ncm-code-doc.git
cd ncm-code-doc
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -e .
```

### 3. Descargar los pesos del modelo
Los pesos adaptados para código están alojados en Hugging Face en el repositorio [`r2d2aparicio/ncm-code-doc`](https://huggingface.co/r2d2aparicio/ncm-code-doc).

Para descargarlos en la ruta requerida (`artifacts/ncm_code_doc/`):
```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='r2d2aparicio/ncm-code-doc', local_dir='artifacts/ncm_code_doc')"
```

El directorio de artefactos debe contener:
- `generator.pt` (~354 MB)
- `tokenizer.pt` (~299 MB)

## Configuración MCP

### Antigravity IDE / Claude Desktop
Agrega la definición del servidor a la configuración de servidores MCP (`mcpServers`):

```json
{
  "mcpServers": {
    "ncm-fast-writer": {
      "command": "/ruta/absoluta/a/ncm-code-doc/.venv/bin/python",
      "args": [
        "-u",
        "/ruta/absoluta/a/ncm-code-doc/ncm_doc_server.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```
*(En Windows, sustituye las rutas por la ubicación de `python.exe` y `ncm_doc_server.py` utilizando barras inclinadas o barras invertidas dobles).*

### Herramienta expuesta

- **`fast_ncm_redaction`**: Genera un borrador o resumen conciso de documentación técnica a partir de un fragmento de código o contexto.
  - Parámetros:
    - `context_or_code` (string): Código fuente o fragmento técnico de entrada.
  - Salida: Texto conciso en lenguaje natural con la descripción funcional del bloque.

## Re-entrenamiento (Fine-Tuning)

El repositorio incluye el dataset `data_train.jsonl` (5,000 pares extraídos de CodeSearchNet).

Para entrenar o adaptar el modelo con nuevos datos:
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python finetune_ncm.py
```

Parámetros clave configurados en `finetune_ncm.py`:
- Longitud de prefijo (código): 512 tokens.
- Longitud objetivo (resumen): 48 tokens.
- Optimización de memoria: Precisión mixta FP16 y acumulación de gradiente (batch efectivo 4) para adaptarse a GPUs de 6 GB VRAM.
- Descongelamiento selectivo del decodificador VQ-VAE para vocabulario técnico de programación.

## Estructura del Proyecto

```
ncm-code-doc/
├── ncm_doc_server.py     # Punto de entrada del servidor MCP (stdio)
├── finetune_ncm.py       # Script de fine-tuning adaptado a pares de código
├── data_train.jsonl      # Dataset de entrenamiento (CodeSearchNet)
├── pyproject.toml        # Empaquetado y dependencias del proyecto
├── models/               # Arquitectura NCM y VQ-VAE jerárquico
├── tokenization/         # Tokenizador BPE y mapeos de vocabulario
├── config/               # Configuraciones Hydra / PyTorch
└── artifacts/            # Checkpoints locales (ignorado en Git)
```

## Referencias y Créditos

Este trabajo adapta y extiende la arquitectura de:
- **Next-Concept Modeling: Scalable Non-Autoregressive Generation via Hierarchical Latent Space**  
  *Stanford University (Hazy Research)*  
  Repositorio original: [HazyResearch/ncm](https://github.com/HazyResearch/ncm)
