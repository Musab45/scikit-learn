# HOG Feature Visualization

This project demonstrates how to extract **Histogram of Oriented Gradients (HOG)** features from an image using `scikit-image` and visualize both the original image and its corresponding HOG feature representation.

## Example Output

![HOG Visualization](https://github.com/Musab45/scikit-learn/raw/main/hog_visualization/hog_visualization_1.png)
![HOG Visualization 2](https://github.com/Musab45/scikit-learn/raw/main/hog_visualization/hog_visualization_2.png)


## How It Works

- The input image is processed using the `hog()` function from `skimage.feature`.
- Gradient orientation histograms are computed over localized regions of the image.
- A visualization image is generated to display the detected edge structures and gradients.
- Both the original image and the HOG visualization are displayed side-by-side using Matplotlib.

## Key Steps in Code

- **HOG Extraction:**

```python
from skimage.feature import hog

fd, hog_image = hog(image, 
                    orientations=8, 
                    pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), 
                    visualize=True, 
                    channel_axis=-1)