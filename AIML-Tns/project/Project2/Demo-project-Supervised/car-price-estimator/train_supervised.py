import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

df = pd.read_csv('used_cars.csv')

X = df[['age_years', 'mileage_km', 'fuel_type']]

y = df['selling_price']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), ['fuel_type'])
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

model = DecisionTreeRegressor(max_depth=6, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(f"Supervised Model Accuracy (R2 Score): {r2_score(y_test, y_pred):.2f}")

joblib.dump(model, 'car_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Model and Preprocessor saved successfully!")