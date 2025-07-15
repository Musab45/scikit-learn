import pandas as pd
import numpy as np
from pandas import crosstab
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans

# -- Step 1: Data Loading and Preparation --

# Load the dataset
wine_data = load_wine()
X = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)
# The 'target' in this dataset is the cultivar (0, 1, 2), which we'll use for clustering analysis later.
# For regression/classification, let's use 'alcohol' to create a quality score.
y_true_quality = X['alcohol'].apply(lambda x: round(x * 0.5 + np.random.normal(0, 0.1))) # Create a plausible quality score
# 1. Regression Target: The numeric quality score itself.
y_reg = y_true_quality
# 2. Classification Target: Create a binary label: 'good' (1) or 'bad' (0).
# Let's define "good" wine as having a quality score > 6.
y_cls = (y_reg > 6).astype(int)
# Display our prepared data
print("Features (X):")
print(X.head())
print("\nRegression Target (y_reg):")
print(y_reg.head())
print("\nClassification Target (y_cls):")
print(y_cls.head())

# -- Step 2: Supervised Learning - Regression (Predicting Quality Score) --

#Spliting data for regression
X_train, X_test, y_train_reg, y_tests_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
#Create regression pipeline
reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LogisticRegression())
])
#Training the model
reg_pipeline.fit(X_train, y_train_reg)
#Make predictions and evaluate
predictions_reg = reg_pipeline.predict(X_test)
mse = mean_squared_error(y_tests_reg, predictions_reg)
print(f'MSE: {mse:.2f}') # avg squared difference b/w the actual and predicted quality scores

# -- Step 3: Supervised Learning - Classification (Good Wine vs Bad Wine) --

#Splitting data for classification
X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)
#Classification pipeline
cls_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])
#Training the model
cls_pipeline.fit(X_train, y_train_cls)
#Making predictions and evaluating
predictions_cls = cls_pipeline.predict(X_test)
accuracy = accuracy_score(y_test_cls, predictions_cls)
precision = precision_score(y_test_cls, predictions_cls)
recall = recall_score(y_test_cls, predictions_cls)
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}, we are correct {precision*100:.0f}% of the time')
print(f'Recall: {recall:.2f}, we have correctly identified {recall*100:.0f}% of all the "good" wines')


# -- Step 4: Unsupervised Learning: Clustering (Discovering Wine Types) --

#No train/test split required for unsupervised learning
#We want to find patterns in the entire dataset
cluster_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clusterer', KMeans(n_clusters=3, random_state=42, n_init=10))
])
#Get the cluster assignments for each wine
discovered_clusters = cluster_pipeline.fit_predict(X)
#Adding the results back to our DataFrame for analysis
analysis_df = X.copy()
analysis_df['true_quality_score'] = y_reg
analysis_df['is_good_wine'] = y_cls
analysis_df['discovered_cluster'] = discovered_clusters
#Interesting Part! Seeing how the clusters align with quality
cluster_quality_analysis = pd.crosstab(analysis_df['discovered_cluster'], analysis_df['is_good_wine'])
print(cluster_quality_analysis)