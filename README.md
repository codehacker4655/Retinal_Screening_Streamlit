# RetinaAI — H3 demo

This package was extracted from the uploaded Kaggle notebook **without retraining**.

## Notebook-derived definitions

- ROI: notebook index 66 (`get_retinal_roi`)
- Final conservative P2 Retinex: notebook index 75 (later override of earlier P2 versions)
- Reflection-padded aspect-preserving resizing and class order: notebook index 78
- P2 validation dataset transformation order: notebook index 107
- Hybrid model: notebook index 161, H3 hyperparameters: index 175
- CNN / Transformer token Grad-CAM: notebook indices 194 and 196

## Setup (VS Code terminal)

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS/Linux: .venv/bin/python -m pip install -r requirements.txt
python -m pip install -r requirements.txt
```

Copy your **actual verified H3 best checkpoint** to `models/best_h3_hybrid.pth`.
Do not substitute a H5-A checkpoint: the architectures differ. The checkpoint is
not included in this ZIP because it was not uploaded in this conversation.

Start:

```bash
python -m streamlit run app.py
```

## Optional LLM narrative

Copy `.env.example` to `.env`; set `OPENAI_API_KEY` if desired. This is optional:
the deterministic report and PDF work without an LLM connection. Never commit `.env`.

## Notes

- Keep model on CPU for local demonstration, if available; first run can be slower.
- The application does NOT use test labels or perform testing automatically.
- Original H3 validation result (previously reproduced in Kaggle): Macro-F1 0.7438.
- Transformer visualization is an adapted gradient-token attribution, not attention rollout,
  and neither heatmap is a clinical lesion segmentation.
- Model scores are not calibrated probabilities. The report is for research only.
- This package has static/compile validation only; without your H3 .pth checkpoint and
  local runtime dependencies, a genuine end-to-end inference test cannot be certified.
