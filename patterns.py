from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

X = [[100],
     [200],
     [300]]

scaler = MinMaxScaler()
scaler.fit(X) # training the model
X_scaled = scaler.transform(X)
# or fit_transform(X) as a shortcut
print(f'Scaled Data(X): {X_scaled}')

# prediction
flower_samples = [[5.1, 3.5, 1.4, 0.2],     # Setosa
                  [6.2, 3.4, 5.4, 2.3]]     # Virginica
flower_categories = [0, 2]


model = LogisticRegression()
model.fit(flower_samples, flower_categories)

new_flower = [[5.0, 3.4, 1.5, 0.2]]
prediction = model.predict(new_flower)

print(f"Flower Categories: {prediction}")