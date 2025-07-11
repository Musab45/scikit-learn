import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from patterns import prediction

# - sample data
data = pd.DataFrame({
    'age': [25, 45, 35, 50, 23],
    'country': ['USA', 'UK', 'USA', 'Canada', 'UK'],
    'monthly_spend': [50, 200, 120, 80, 40],
    'subscribed': [0, 1, 1, 0, 0]
})

# - separating feature
# input feature (what the model will use)
X = data.drop('subscribed', axis=1)
# output (what the model will predict)
y = data['subscribed']

# - specifying column types
numeric_features = ['age', 'monthly_spend'] # to be scaled
categorical_features = ['country'] # to be encoded

# - preprocessing pipeline using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# - full pipeline (Preprocessing + Model)
pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ]
)

# - split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# - training the pipeline
pipeline.fit(X_train, y_train) # trains logistic regression model on the processed data

# - make predictions
predictions = pipeline.predict(X_test)

# - evaluating the model
accuracy = pipeline.score(X_test, y_test)
print(f'Model Accuracy: {accuracy}')




#  -- Summary Diagram --
# Raw Data (X, y)
#      |
# Train/Test Split
#      |
# Pipeline
#     - Preprocess (scale, encode)
#     - Train (logistics regression)
#      |
# Predict and Score