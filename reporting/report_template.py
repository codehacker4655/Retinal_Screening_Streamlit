"""Deterministic fallback: no unsupported clinical assertions."""

def make_report_text(result, image_id):
    category=result["predicted_class"]
    scores="; ".join(f"{name}: {value:.1%}" for name,value in result["probabilities"].items())
    return (f"RETINAL SCREENING RESEARCH REPORT\n\n"
            f"Image identifier: {image_id}\nModel: {result['model']}\n"
            f"Predicted category: {category}\nModel-assigned score: "
            f"{result['probabilities'][category]:.1%}\n\n"
            f"Scores across all categories: {scores}\n\n"
            "Method: Retinal ROI extraction, conservative P2 Retinex, "
            "aspect-preserving 260x260 resize, ImageNet normalization, hybrid inference.\n"
            "Explainability: CNN Grad-CAM and Transformer token-feature attribution are "
            "provided as illustrative figures, not lesion-localization evidence.\n\n"
            "Clinical limitation: This is an unvalidated research prototype. "
            "Model probabilities are not calibrated diagnostic confidence. "
            "No disease severity, lesion location, treatment or diagnosis is established. "
            "A qualified eye-care professional must interpret any screening result.")
