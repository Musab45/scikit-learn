import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

data = pd.DataFrame({
    'income': [30000, 50000, 40000, 80000, 20000, 60000, 55000],
    'age': [22, 35, 28, 45, 20, 40, 32],
    'car_type': ['sedan', 'SUV', 'sedan', 'SUV', 'hatchback', 'SUV', 'hatchback'],
    'approved': [0, 1, 1, 1, 0, 1, 1]  # 1 = Approved, 0 = Not approved
})

X = data.drop('approved', axis=1)
y = data['approved']

numeric_features = ['income', 'age']
categorical_features = ['car_type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=3))
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)

print('Predictions: ', predictions)
print(f'Model Accuracy: {accuracy:.2f}')

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    X_train_transformed[:, 0], X_train_transformed[:, 1],
    c=y_train, cmap='bwr', edgecolor='k', label='Train', marker='o'
)

plt.scatter(
    X_test_transformed[:, 0], X_test_transformed[:, 1],
    c=predictions, cmap='bwr', edgecolor='k', label='Test (Predicted)', marker='s', s=100
)

plt.xlabel('Scaled Income')
plt.ylabel('Scaled Age')
plt.title('KNN Loan Approval Prediction (2D)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()