import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

# Sample student data
data = pd.DataFrame({
    'study_hours': [1, 5, 3, 8, 2, 6, 4],
    'country': ['Pakistan', 'USA', 'USA', 'Pakistan', 'Canada', 'USA', 'Canada'],
    'passed': [0, 1, 0, 1, 0, 1, 1]  # 1 = Passed, 0 = Failed
})

# Splitting into input features and target
X = data.drop('passed', axis=1)
y = data['passed']

# Defining categories
numeric_features = ['study_hours']
categorical_features = ['country']

# Creating a preprocessing transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Full pipeline: preprocessing + model
pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ]
)

# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training the pipeline
pipeline.fit(X_train, y_train)

# Making predictions on test data
predictions = pipeline.predict(X_test)

# Model evaluation
accuracy = pipeline.score(X_test, y_test)

print('Predictions: ', predictions)
print(f'Model Accuracy: {accuracy:.2f}')