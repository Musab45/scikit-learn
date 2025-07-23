import os
import shutil
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import io, color, transform

CLASSES = ['rose', 'sunflower']
NUM_IMAGES_PER_CLASS = 50
DATA_DIR = '../flower_dataset'


# This function is for synthetic data generation and is not used here.
# def generate_synthetic_images(): ...

def process_and_load_images():
    features = []
    labels = []
    # Define a consistent size for all images
    image_size = (128, 128)

    for flower_class in CLASSES:
        class_dir = os.path.join(DATA_DIR, flower_class)
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory not found, skipping: {class_dir}")
            continue
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(class_dir, img_name)

            img = io.imread(img_path)

            img_resized = transform.resize(img, image_size)

            img_gray = color.rgb2gray(img_resized)

            hog_features = hog(img_gray,
                               pixels_per_cell=(8, 8),
                               cells_per_block=(2, 2),
                               visualize=False,
                               channel_axis=None)

            features.append(hog_features)
            labels.append(flower_class)

    df = pd.DataFrame(features)
    df['label'] = labels
    return df


if __name__ == '__main__':
    # generate_synthetic_images()
    flower_dataset = process_and_load_images()

    if flower_dataset.empty:
        print("Error: No data was loaded. Check your DATA_DIR path and image files.")
    else:
        print('\nDataFrame Head:')
        print(flower_dataset.head())

        X = flower_dataset.drop('label', axis=1)
        y = flower_dataset['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classification', LogisticRegression(solver='liblinear', C=1.0))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'\nAccuracy: {accuracy:.2f}')
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))

        try:
            test_img_path = 'test_2.jpeg'
            test_img = io.imread(test_img_path)

            test_image_size = (128, 128)
            test_img_resized = transform.resize(test_img, test_image_size)
            test_img_gray = color.rgb2gray(test_img_resized)

            single_image_features = hog(test_img_gray,
                                        pixels_per_cell=(8, 8),
                                        cells_per_block=(2, 2),
                                        visualize=False,
                                        channel_axis=None)

            prediction = pipeline.predict([single_image_features])
            print(f'\nPredicted flower for {test_img_path}: {prediction[0]}')

        except FileNotFoundError:
            print(f"\nTest image not found at '{test_img_path}'. Skipping custom test.")
        except Exception as e:
            print(f"\nAn error occurred during custom testing: {e}")
