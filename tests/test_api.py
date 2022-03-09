from fastapi import FastAPI
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"greeting": "Hello World!"}


def test_predict_salary_above():
    data = {
        "age": 34,
        "workclass": "Local-gov",
        "fnlgt": 280464,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Protective-serv",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == "Predicted salary is greater than 50K."


def test_predict_salary_less():
    data = {
        "age": 48,
        "workclass": "Private",
        "fnlgt": 171095,
        "education": "Assoc-acdm",
        "education_num": 12,
        "marital_status": "Divorced",
        "occupation": "Exec-managerial",
        "relationship": "Unmarried",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "England"
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert response.json() == "Predicted salary is less than 50K."