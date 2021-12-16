from fastapi import FastAPI
import model

app = FastAPI()


@app.get('/')
async def home():
    return {"hello": "aziz"}


@app.get('/predict')
async def predict():
    return {"test_predictions": list(model.predict('seoul-bike-rental-ai-pro-iti/test.csv'))}

