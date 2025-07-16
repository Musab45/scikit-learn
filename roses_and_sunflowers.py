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

CLASSES = ['rose', 'sunflower']
NUM_IMAGES_PER_CLASS = 50
DATA_DIR = 'flower_dataset'

def generate_synthetic_images():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)
    for flower_class in CLASSES:
        class_dir = os.path.join(DATA_DIR, flower_class)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(NUM_IMAGES_PER_CLASS):
            img = Image.new('RGB', (64,64), 'black')
            draw = ImageDraw.Draw(img)
            if flower_class == 'rose':
                draw.ellipse([(10,10), (54, 54)], fill='red', outline='red')
            else:
                draw.rectangle([(10, 10), (54, 54)], fill='yellow', outline='yellow')
            img.save(os.path.join(class_dir, f'{flower_class}_{i}.png'))

def process_and_load_images():
    features = []
    labels = []
    for flower_class in CLASSES:
        class_dir = os.path.join(DATA_DIR, flower_class)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = Image.open(img_path).convert('L')
            img = img.resize((64,64))
            img_vector = np.array(img).flatten()
            features.append(img_vector)
            labels.append(flower_class)
    df = pd.DataFrame(features)
    df['label'] = labels
    return df

if __name__ == '__main__':
    generate_synthetic_images()
    flower_dataset = process_and_load_images()
    print('\nDataFrame Head:')
    print(flower_dataset.head())

    X = flower_dataset.drop('label', axis=1)
    y = flower_dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classification', LogisticRegression(solver='liblinear'))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print(classification_report(y_test, y_pred))

    # Custom Testing
    test_img = Image.new('RGB', (64, 64), 'black')
    draw = ImageDraw.Draw(test_img)
    # Create synthetic image here
    draw.rectangle([(10, 10), (54, 54)], fill='#FFD700', outline='#FFD700')
    test_img.save('test.png')
    test_img = Image.open('test.png').convert('L')
    test_img = test_img.resize((64, 64))
    test_img_vector = np.array(test_img).flatten()


    single_image_features = [test_img_vector]
    single_image_label = y_test.iloc[20]

    prediction = pipeline.predict(single_image_features)[0]

    print(f'Predicted flower: {prediction}')
