from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import numpy as np


def create_ensemble_model():
    models = [
        ('MLP', MLPRegressor(hidden_layer_sizes=(300, 10), solver="adam", activation="relu", learning_rate_init=0.001, max_iter=10000, random_state=42)),
        ('RF', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('GB', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ('SVR', SVR(kernel='rbf'))
    ]
    return models

def train_ensemble(X, y, models):
    for name, model in models:
        model.fit(X, y)
    return models

def predict_ensemble(X, models):
    predictions = []
    for name, model in models:
        pred = model.predict(X)
        predictions.append(pred)
    return np.mean(predictions, axis=0)
