# Put the code for your API here.
# Put the code for your API here.

from joblib import load
import os
from typing import Dict
from sys import exit

from fastapi import FastAPI
import pandas as pd
import numpy as np

from api.schemas import Census
from src.ml.data import process_data
from src.ml.model import inference

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

model = load("./model/model.joblib")
encoder = load("./model/encoder.joblib")
lb = load("./model/lb.joblib")
app = FastAPI()


@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}


@app.post("/predict")
def predict_salary(data: Census):

    input_data = data.dict()

    age = input_data["age"]
    workclass = input_data["workclass"]
    fnlgt = input_data["fnlgt"]
    education = input_data["education"]
    education_num = input_data["education_num"]
    marital_status = input_data["marital_status"]
    occupation = input_data["occupation"]
    relationship = input_data["relationship"]
    race = input_data["race"]
    sex = input_data["sex"]
    capital_gain = input_data["capital_gain"]
    capital_loss = input_data["capital_loss"]
    hours_per_week = input_data["hours_per_week"]
    native_country = input_data["native_country"]

    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    input_array = np.array([[
        age,
        workclass,
        fnlgt,
        education,
        education_num,
        marital_status,
        occupation,
        relationship,
        race,
        sex,
        capital_gain,
        capital_loss,
        hours_per_week,
        native_country
    ]])

    df_input = pd.DataFrame(data=input_array, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country"
    ])
    input_x, _, _, _ = process_data(
        df_input, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    prediction = inference(model, input_x)
    if prediction[0]:
        prediction = "Predicted salary is greater than 50K."
    else:
        prediction = "Predicted salary is less than 50K."

    return prediction