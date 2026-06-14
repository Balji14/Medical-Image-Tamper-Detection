"""
Grad-CAM adapted for HResUNet++ segmentation.
Hooks the last encoder conv block (conv4_0) and produces
a spatial attention heatmap over the input image.
"""
import numpy as np
import cv2
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._activations = None
        self._gradients = None

        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self._activations = out.detach()

    def _bwd_hook(self, module, grad_in, grad_out):
        self._gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_size=None) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)           # (1,1,H,W)
        self.model.zero_grad()
        output.mean().backward()

        grads = self._gradients                     # (1,C,h,w)
        acts  = self._activations                   # (1,C,h,w)

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1,1,h,w)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()

        h, w = target_size if target_size else (input_tensor.shape[2], input_tensor.shape[3])
        cam = cv2.resize(cam, (w, h))
        return cam.astype(np.float32)


def overlay_heatmap(gray_img: np.ndarray, cam: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """
    Blend Grad-CAM heatmap onto a greyscale image.
    Returns uint8 RGB image.
    """
    img_u8 = (gray_img * 255).astype(np.uint8) if gray_img.max() <= 1.0 else gray_img.astype(np.uint8)
    bgr    = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)

    heat_u8 = (cam * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    blended = cv2.addWeighted(bgr, 1 - alpha, heat_color, alpha, 0)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
