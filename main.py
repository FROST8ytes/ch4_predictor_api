import torch
import torch.nn as nn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


class CH4PredictionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)


def load_model(model_path, **kargs):
    # Load the saved model data
    checkpoint = torch.load(model_path, **kargs)

    # Create a new model instance
    model = CH4PredictionModel(len(checkpoint["input_features"]))

    # Load the state dictionary
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint["input_features"], checkpoint["scaler_X"], checkpoint["scaler_y"]


def predict_ch4_emission(params_dict, model, features, scaler_X, scaler_y):
    # Create a DataFrame with the input parameters
    input_df = pd.DataFrame([params_dict])

    # Ensure all features are present in the correct order
    input_df = input_df.reindex(columns=features)

    # Scale the input
    input_scaled = scaler_X.transform(input_df)

    # Convert to tensor
    input_tensor = torch.FloatTensor(input_scaled)

    # Set model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        prediction_scaled = model(input_tensor)

    # Inverse transform the prediction
    prediction = scaler_y.inverse_transform(prediction_scaled.numpy())

    return prediction[0][0]


# Load the model
model, features, scaler_X, scaler_y = load_model(
    "trained_models/ch4_prediction_model.pth", weights_only=False)


class Params(BaseModel):
    t: float
    precipitation: float
    soc: float
    tn: float
    ph: float
    bd: float
    clay: float
    growth_period: float
    n_amount: float
    p2o5_amount: float
    k2o_amount: float


@app.post("/echo")
def echo(params: Params):
    try:
        params_dict = params.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return params


@app.post("/predict")
def predict(params: Params):
    try:
        params_dict = params.model_dump()
        predicted_emission = predict_ch4_emission(
            params_dict, model, features, scaler_X, scaler_y)
        return {"predicted_emission": predicted_emission.item()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    params = Params(t=17.0, precipitation=1400, soc=20, tn=2.0, ph=6.5, bd=1.5,
                    clay=25, growth_period=100, n_amount=150, p2o5_amount=75, k2o_amount=100)

    print(predict(params))

# To run the FastAPI app, use the command: uvicorn main:app --reload
