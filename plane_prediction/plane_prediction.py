import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.pipeline import Pipeline
from skimage import io, color, transform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


CLASSES = ['f22', 'j10', 'su35']
NUM_OF_IMAGES_PER_CLASS = 200
DATA_DIR = 'data'

def load_and_process_images():
    features = []
    labels = []

    img_size = (128, 128)

    for animal in CLASSES:
        class_dir = os.path.join(DATA_DIR, animal)
        if not os.path.isdir(class_dir):
            print(f'{class_dir} is not a valid path')
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', 'jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_dir, img_name)
            img = io.imread(img_path)
            img_resized = transform.resize(img, img_size)
            img_gray = color.rgb2gray(img_resized)

            hog_features = hog(
                img_gray,
                orientations = 20,
                pixels_per_cell = (16, 16),
                cells_per_block = (2, 2),
                block_norm='L2-Hys',
                visualize = False,
                channel_axis=None
            )
            features.append(hog_features)
            labels.append(animal)
    df = pd.DataFrame(features)
    df['label'] = labels
    return df

if __name__ == '__main__':
    dataset_df = load_and_process_images()

    if dataset_df.empty:
        print('Dataset is empty')
    else:
        X = dataset_df.drop('label', axis=1)
        y= dataset_df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2 , random_state = 42, stratify = y)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', probability=True, random_state=42))
        ])

        pipeline.fit(X_train, y_train)
        print("Model training complete.")

        # Step 4: Evaluate the Model against client requirements
        print("\n--- Model Evaluation ---")
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")

        if accuracy >= 0.95:
            print("✅ Performance Requirement Met (>= 95% Accuracy)")
        else:
            print("❌ Performance Requirement NOT Met (< 95% Accuracy)")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=CLASSES))

        MODEL_FILENAME = 'plane_prediction.joblib'
        print(f"\nSaving the final trained model to '{MODEL_FILENAME}'...")
        joblib.dump(pipeline, MODEL_FILENAME)
        print("✅ Model saved successfully.")