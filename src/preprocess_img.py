"""Modulo encargado del preprocesamiento de imagenes antes de la prediccion."""

import cv2
import numpy as np


def preprocess(array):
    """Deja una imagen lista para ser evaluada por el modelo de prediccion.

    Redimensiona la imagen a 512x512, la convierte a escala de grises,
    le aplica la tecnica CLAHE (mejora de contraste local) y normaliza
    sus valores para que queden entre 0 y 1. Finalmente le agrega las
    dimensiones adicionales que Keras espera al recibir una imagen.

    Args:
        array: arreglo NumPy de la imagen a preprocesar (salida de
            read_dicom_file o read_jpg_file).

    Returns:
        Un arreglo NumPy con forma (1, 512, 512, 1), listo para
        pasarse directamente al modelo.
    """
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    return array