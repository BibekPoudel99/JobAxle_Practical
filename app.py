from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import predict as pred

app = FastAPI(
    title="Car price prediction API",
)

class CarInput(BaseModel):
    model_year: int
    milage: float
    fuel_type: str  # e.g., "Gasoline", "Hybrid"
    transmission: str  # e.g., "8-Speed Automatic"
    ext_col: str  # External color
    int_col: str  # Internal color
    clean_title: str  # "Yes" or "No"
    is_accident: str  # "YES" or "NO"
    brand: str
    horsepower: Optional[float] = 0.0
    liters: Optional[float] = 0.0

    class Config:
        example = {
            "model_year": 2020,
            "milage": 50000.0,
            "fuel_type": "Gasoline",
            "transmission": "8-Speed Automatic",
            "ext_col": "Black",
            "int_col": "Gray",
            "clean_title": "Yes",
            "is_accident": "NO",
            "brand": "Toyota",
            "horsepower": 300.0,
            "liters": 3.5
        }

@app.get("/")
def read_root():
    return {
        "message": "Welcome to Car Price Prediction API",
        "endpoints": {
            "linear_regression": "/predict/linear",
        }
    }

@app.get("/health")
def health_check():
    return {"status": " API is running"}

# Linear Regression prediction
@app.post("/predict/linear")
def predict_linear(car: CarInput):
    """Predict exact car price using Linear Regression"""
    try:
        features = pred.prepare_features(car.dict())
        if features is None:
            raise HTTPException(status_code=400, detail="Error preparing features")
        
        price = pred.predict_linear_regression(features)
        
        return {
            "model": "Linear Regression",
            "predicted_price": f"${price:,.2f}",
            "price_value": price
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)