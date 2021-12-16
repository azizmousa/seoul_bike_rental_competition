import os
import datetime
from fastapi import FastAPI, UploadFile, File
import model
from model import PredictionItem
app = FastAPI()


@app.get('/')
async def home():
    return {"hello": "aziz"}


@app.get('/predict_test/')
async def predict_test():
    return {"test_predictions": model.predict_file('seoul-bike-rental-ai-pro-iti/test.csv')}


@app.post("/predict_file/")
async def predict_file(file: UploadFile = File(...)):
    os.makedirs('uploaded', exist_ok=True)
    file_name = str(datetime.datetime.now()) + f"_{file.filename}"
    file_path = f"uploaded/{file_name}"
    with open(file_path, 'wb') as out_file:
        content = await file.read()
        out_file.write(content)
    return {"file_prediction": model.predict_file(file_path)}


@app.post("/predict_single/")
async def predict_single(item: PredictionItem):
    return {"single_prediction": model.predict_single(item.p_id, item.date, item.hour, item.temperature, item.humidity,
                                                      item.wind_speed, item.visibility, item.dew_point, item.solar_rad,
                                                      item.rain_fall, item.snow_fall, item.season, item.holiday,
                                                      item.functioning_day)}
