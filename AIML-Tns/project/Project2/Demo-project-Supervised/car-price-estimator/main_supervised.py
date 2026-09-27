from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(
    title="Supervised Car Price Estimator API", 
    description="FastAPI Backend for Decision Tree Model"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    model = joblib.load('car_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
except Exception as e:
    raise RuntimeError(f"Error loading model files: {e}")

class CarSpecs(BaseModel):
    age_years: int = Field(..., ge=0, le=30, description="Vehicle age in years (between 0 and 30)")
    mileage_km: float = Field(..., ge=0, description="Total distance driven in KM (must be non-negative)")
    fuel_type: str = Field(..., description="Fuel type: Petrol, Diesel, or Electric")

@app.get("/")
def home():
    return {"status": "active", "message": "Supervised Car Price Estimator API"}

@app.post("/predict")
def predict_price(data: CarSpecs):
    try:
        input_df = pd.DataFrame([{
            'age_years': data.age_years,
            'mileage_km': data.mileage_km,
            'fuel_type': data.fuel_type
        }])
        
        processed_input = preprocessor.transform(input_df)
        
        predicted_price = float(model.predict(processed_input)[0])
        
        return {
            "estimated_price_usd": round(predicted_price, 2),
            "vehicle_specs": data.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))