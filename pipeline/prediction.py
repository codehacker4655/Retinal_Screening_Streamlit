"""Checkpoint loading and five-class inference. No training or downloads."""
from pathlib import Path
import torch
import torch.nn.functional as F
from models.hybrid_model import HybridEfficientNetViT
from pipeline.preprocessing import CLASS_NAMES


def load_h3(path, device="cpu"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"H3 checkpoint not found: {path}. Copy the saved .pth file into models/.")
    model = HybridEfficientNetViT(num_classes=5, embed_dim=128, num_heads=4,
                                  transformer_layers=1, dropout=0.3,
                                  pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict(model, tensor, device="cpu"):
    model.eval()
    with torch.inference_mode():
        logits = model(tensor.to(device))
        probs = F.softmax(logits, dim=1)[0].cpu().tolist()
    idx = int(max(range(5), key=lambda i: probs[i]))
    return {"predicted_class": CLASS_NAMES[idx],
            "predicted_index": idx,
            "probabilities": dict(zip(CLASS_NAMES, [float(x) for x in probs])),
            "model": "H3 EfficientNet-B2 + custom Transformer",
            "validation_macro_f1": 0.7438}
