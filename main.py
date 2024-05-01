from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd


class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI(
    title="Proyecto de MLOPS",
    description="Predicción de precio de casas en King County",
    version="0.0.1"
)


@app.post("/predict-price/")
async def predict_house_price(house_data: HouseInput):
    input_data = pd.DataFrame([house_data.dict()])
    predicted_price = model.predict(input_data)

    return {"El precio predecido es": predicted_price[0]}