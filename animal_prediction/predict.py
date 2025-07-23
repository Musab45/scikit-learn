# For prediction new pokeball images using the joblib file
import os
from skimage import io, color, transform
import numpy as np
from PIL import Image
from skimage.feature import hog
import joblib

# --- 1. CONFIGURATION ---

# Path to your saved model file from the training script
MODEL_PATH = 'animal_prediction.joblib'

# The same image size used during training
IMAGE_SIZE = (128, 128)


# --- 2. PREDICTION FUNCTION ---

def predict_pokeball_type(image_path, model):
    """
    Loads a single image, applies the same HOG preprocessing as the training
    script, and uses the loaded model to predict its class.

    Args:
        image_path (str): The full path to the new image file.
        model: The loaded Scikit-learn pipeline/model object.

    Returns:
        A tuple containing the predicted class name (str) and the
        probabilities for each class (numpy.ndarray). Returns (None, None) on error.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image path not found at '{image_path}'")
        return None, None

    try:
        # Step A: Load and preprocess the image (must match training)
        img = io.imread(image_path)
        img_resized = transform.resize(img, IMAGE_SIZE)
        img_gray = color.rgb2gray(img_resized)

        image_array = np.array(img_gray)

        # Step B: Extract HOG features (parameters must match training)
        # These parameters are copied from your training script.
        hog_features = hog(image_array,
                           orientations=9,
                           pixels_per_cell=(8,8),
                           cells_per_block=(2, 2),
                           visualize=False)

        # The model expects a 2D array, so we reshape our 1D feature vector
        features_2d = hog_features.reshape(1, -1)

        # Step C: Make the prediction
        prediction = model.predict(features_2d)
        probabilities = model.predict_proba(features_2d)

        return prediction[0], probabilities[0]

    except Exception as e:
        print(f"An error occurred while processing the image: {e}")
        return None, None


# --- 3. MAIN EXECUTION BLOCK ---

if __name__ == '__main__':
    # Step 1: Load the trained model from the file
    print(f"Loading model from '{MODEL_PATH}'...")
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Please run the training script first.")
    else:
        loaded_model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully.")

        # --- Make a Prediction ---
        TEST_IMAGE_PATH = 'tests/dog_test.png'

        print(f"\nPredicting type for image: '{TEST_IMAGE_PATH}'...")
        predicted_class, class_probabilities = predict_pokeball_type(TEST_IMAGE_PATH, loaded_model)

        if predicted_class is not None:
            print(f"\n---> Predicted Class: {predicted_class.upper()} <---")

            # Display the probabilities for each class
            print("\nConfidence Scores:")
            # Creates a nice display of class names and their predicted probabilities
            for i, class_name in enumerate(loaded_model.classes_):
                print(f"  - {class_name.capitalize()}: {class_probabilities[i]:.2%}")

