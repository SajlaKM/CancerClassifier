import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_segmentation_mask(image):
    """
    Generates a pseudo-segmentation mask using thresholding and edge detection.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's thresholding
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to refine mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask

# Filter only 'pituitary' images
filtered_images = []
filtered_labels = []
filtered_masks = []

for i in range(len(images)):
    if labels[i] in ["pituitary"]:  # ✅ Select only specific classes
        filtered_images.append(images[i])
        filtered_labels.append(labels[i])
        filtered_masks.append(generate_segmentation_mask((images[i] * 255).astype(np.uint8)))

# Number of filtered samples
num_samples = 5

# Plot original images and corresponding masks
fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))

for i in range(num_samples):
    # Show MRI image
    axes[0, i].imshow(filtered_images[i])
    axes[0, i].set_title(f"Class: {filtered_labels[i]}")
    axes[0, i].axis("off")

    # Show corresponding mask
    axes[1, i].imshow(filtered_masks[i], cmap="gray")
    axes[1, i].set_title("Segmentation Mask")
    axes[1, i].axis("off")

plt.tight_layout()
plt.show()
