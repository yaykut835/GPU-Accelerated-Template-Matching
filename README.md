# GPU-Accelerated-Template-Matching
A high performance Python utility for performing Normalized cross correlation template matching using pytorch gpu kernels. This module provides a significant speedup over standard cpu based methods (like cv2.matchTemplate) while maintaining a seamless fallback mechanism.

Gpu Acceleration: Leverages PyTorch's conv2d and avg_pool2d for massive parallelization.
Ncc algorithm: Robust matching that handles lighting and contrast variations.
Automatic fallback: Gracefully switches to cv2.matchTemplate(cpu) if a gpu is unavailable or out of memory.

Ensure you have the following installed:
pip install torch numpy opencv-python

Usage
The module is designed to be used as a singleton for efficiency.

import cv2
from gpu_status import get_gpu_instance

# Initialize the gpu handler
gpu_handler = get_gpu_instance()

# Load your images (grayscale)
screen = cv2.imread('screen.png', 0)
template = cv2.imread('template.png', 0)

# Perform matching
# Returns a score map (heatmap)
result = gpu_handler.gpu_template_match(screen, template)

# Locate the best match
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)




How It Works
The core logic transforms the template matching problem into a 2D Convolution operation:Normalization: Both the template and local screen patches are normalized ($z$-score) to ensure the match is independent of brightness.Sliding Window: avg_pool2d is used to calculate local means and standard deviations across the entire screen in a single pass.Cross-Correlation: The normalized template is used as a convolutional kernel to find the highest similarity scores.

print(f"Best Match Score: {max_val:.4f}")
print(f"Coordinates (X, Y): {max_loc}")
