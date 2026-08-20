"""Modulo encargado de orquestar el pipeline completo de prediccion."""

import numpy as np

from src.preprocess_img import preprocess
from src.load_model import model_fun
from src.grad_cam import grad_cam


def predict(array):
    """Orquesta el pipeline completo: preprocesa, predice la clase y genera el heatmap."""
    batch_array_img = preprocess(array)
    model = model_fun()
    preds = model.predict(batch_array_img)
    prediction = np.argmax(preds)
    proba = np.max(preds) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    heatmap = grad_cam(array)
    return (label, proba, heatmap)