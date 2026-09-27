import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 800

age_years = np.random.randint(1, 15, n_samples)

mileage_km = age_years * np.random.randint(8000, 18000, n_samples)

fuel_types = np.random.choice(['Petrol', 'Diesel', 'Electric'], size=n_samples)

base_price = 30000

price = base_price - (age_years * 1200) - (mileage_km * 0.05)

fuel_bonus = np.where(fuel_types == 'Electric', 4000, np.where(fuel_types == 'Diesel', 1500, 0))

final_price = price + fuel_bonus + np.random.normal(0, 1000, n_samples)

final_price = np.clip(final_price, 2000, 45000)

df = pd.DataFrame({
    'age_years': age_years,
    'mileage_km': mileage_km,
    'fuel_type': fuel_types,
    'selling_price': np.round(final_price, 2)
})

df.to_csv('used_cars.csv', index=False)

print("Supervised dataset created successfully: used_cars.csv")