import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import numpy as np

# This script shows how to process and, more importantly, visualize
# Histogram of Oriented Gradients (HOG) features for an image.
# The visualization shows how HOG captures shape and texture
# by showing the direction of intensity changes (gradients) in the image.

image = data.astronaut()
print(f"Image loaded. Shape: {image.shape}")

# HOG Features and Visualization
fd, hog_image = hog(image,
                    orientations=8,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1),
                    visualize=True,
                    channel_axis=-1) # for colored image

print(f"HOG feature vector shape: {fd.shape}")
print(f"HOG visualization image shape: {hog_image.shape}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

# original image
ax1.axis('off')
ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input Image')

# rescale
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# display
ax2.axis('off')
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG')

plt.show()