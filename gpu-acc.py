import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional

class GPUStatus:
    #detects gpu availability and performs gpu accelerated template matching

    def __init__(self):
        self.available = False
        self.name: Optional[str] = None
        self.memory_gb: Optional[float] = None
        self.cuda_version: Optional[str] = None
        self._device = None
        self._probe_gpu()

    def _probe_gpu(self):
        try:
            if torch.cuda.is_available():
                self.available = True
                self.name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                self.memory_gb = round(props.total_memory / 1e9, 2)
                self.cuda_version = torch.version.cuda
                self._device = torch.device("cuda")
        except Exception:
            self.available = False

    def gpu_template_match(self, screen_gray: np.ndarray,
                           template_gray: np.ndarray) -> np.ndarray:

        #performs normalized cross correlation on gpu
        #falls back to cpu on error or if gpu is unavailable.

        if not self.available or self._device is None:
            return cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        try:
            # transfer to gpu
            s = torch.from_numpy(screen_gray.astype(np.float32)).to(self._device)
            t = torch.from_numpy(template_gray.astype(np.float32)).to(self._device)

            th, tw = t.shape

            # template normalization
            t_mean = t.mean()
            t_std = t.std() + 1e-6
            t_norm = (t - t_mean) / t_std

            # reshape to nchw (1, 1, H, W)
            s4 = s.unsqueeze(0).unsqueeze(0)
            t4 = t_norm.unsqueeze(0).unsqueeze(0)

            # local statistics for screen
            kernel_size = (th, tw)
            s_mean = F.avg_pool2d(s4, kernel_size, stride=1, padding=0)
            s_sq_mean = F.avg_pool2d(s4 ** 2, kernel_size, stride=1, padding=0)
            s_std = (s_sq_mean - s_mean ** 2).clamp(min=0).sqrt() + 1e-6

            # screen normalization
            s_norm = (s4 - s_mean) / s_std

            # compute ncc via 2d convolution
            ncc = F.conv2d(s_norm, t4 / (th * tw), padding=0)

            return ncc.squeeze().cpu().numpy()

        except Exception:
            return cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)

# singleton instance
_GPU: Optional['GPUStatus'] = None

def get_gpu_instance() -> 'GPUStatus':
    global _GPU
    if _GPU is None:
        _GPU = GPUStatus()
    return _GPU