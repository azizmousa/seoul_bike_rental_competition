import os
import datetime
from fastapi import FastAPI, UploadFile, File
import model

app = FastAPI()


@app.get('/')
async def home():
    return {"hello": "aziz"}


@app.get('/predict_test')
async def predict_test():
    return {"test_predictions": list(model.predict('seoul-bike-rental-ai-pro-iti/test.csv'))}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    os.makedirs('uploaded', exist_ok=True)
    file_name = str(datetime.datetime.now()) + f"_{file.filename}"
    file_path = f"uploaded/{file_name}"
    with open(file_path, 'wb') as out_file:
        content = await file.read()
        out_file.write(content)
    return {"predictions": list(model.predict(file_path))}


