# Script to train machine learning model.

# import json
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump


from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference

# Add code to load in the data.

# if __name__ == "main":
# Load in the data.
print('Loading data...')
data = pd.read_csv("census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
print('Process train and test data..')
train, test = train_test_split(data, test_size=0.20, random_state=42)

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
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)
# Train and save a model.
print('Train and save model...')
model = train_model(X_train, y_train, hp=True)

dump(model, "./model/model.joblib")
dump(encoder, "./model/encoder.joblib")
dump(lb, "./model/lb.joblib")

# Performance overall
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

