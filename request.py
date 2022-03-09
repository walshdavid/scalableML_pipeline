import requests

url = "https://scalable-ml-pipeline.herokuapp.com/predict"

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
}

data = {
  "age": 42,
  "workclass": "Private",
  "fnlgt": 52789,
  "education": "Masters",
  "education_num": 17,
  "marital_status": "Married",
  "occupation": "Data-Scientist",
  "relationship": "Husband",
  "race": "White",
  "sex": "Male",
  "capital_gain": 4174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
response = requests.post(url, headers=headers, data=data)

status_code = response.status_code
data = response.json()

print(status_code)
print(data)