"""CNN feature Grad-CAM and token-feature attribution for the actual H3 output."""
import numpy as np
import torch
import torch.nn.functional as F
import cv2

class _ActivationCapture:
    def __init__(self, layer):
        self.activation = self.gradient = None
        self.handle = layer.register_forward_hook(self._forward)
    def _forward(self, module, inputs, output):
        self.activation = output.detach()
        if output.requires_grad:
            output.register_hook(self._backward)
    def _backward(self, grad):
        self.gradient = grad.detach()
    def close(self):
        self.handle.remove()

def _normalize(cam, size):
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)[0,0]
    cam = cam - cam.min()
    mx = cam.max()
    return (cam / mx if mx.item() > 1e-8 else torch.zeros_like(cam)).cpu().numpy()

def make_gradcams(model, x, class_index=None):
    """Return two attribution maps for the SAME final hybrid class logit.

    Transformer version is token-feature Grad-CAM adaptation, not attention rollout.
    """
    model.eval()
    x = x.to(next(model.parameters()).device)
    a = _ActivationCapture(model.cnn[-1])
    b = _ActivationCapture(model.transformer_norm)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            if class_index is None:
                class_index = int(logits.argmax(1).item())
            logits[0,class_index].backward()
        assert a.activation is not None and a.gradient is not None
        assert b.activation is not None and b.gradient is not None
        w = a.gradient.mean(dim=(2,3),keepdim=True)
        conv = F.relu((w*a.activation).sum(dim=1,keepdim=True))
        cnn_map = _normalize(conv, x.shape[-2:])
        act,tgrad = b.activation,b.gradient
        assert act.shape[1] == 49
        tw = tgrad.mean(dim=1,keepdim=True)
        token = F.relu((tw*act).sum(dim=2)).reshape(1,1,7,7)
        transformer_map = _normalize(token,x.shape[-2:])
        return cnn_map,transformer_map
    finally:
        a.close();b.close()

def overlay(image, cam, mask=None):
    rgb=np.asarray(image.convert("RGB"),dtype=np.uint8)
    color=cv2.applyColorMap(np.uint8(np.clip(cam,0,1)*255),cv2.COLORMAP_JET)
    color=cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
    blended=cv2.addWeighted(rgb,0.65,color,0.35,0)
    if mask is not None:
        # Mask is a visualization choice; do not change the saved raw attribution.
        blended[~mask]=rgb[~mask]
    from PIL import Image
    return Image.fromarray(blended)
