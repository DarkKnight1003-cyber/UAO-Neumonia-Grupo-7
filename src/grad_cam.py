"""Modulo encargado de generar el mapa de calor Grad-CAM sobre la radiografia."""

import cv2
import numpy as np
import tensorflow as tf

from src.preprocess_img import preprocess
from src.load_model import model_fun


def grad_cam(array):
    """Clasifica la radiografia y genera su mapa de calor Grad-CAM.

    Corre el modelo sobre la imagen preprocesada para obtener la clase
    predicha y su probabilidad, y ademas calcula el gradiente de esa
    prediccion respecto a la ultima capa convolucional (conv10_thisone)
    usando tf.GradientTape, para generar un mapa de calor que muestra
    que zonas de la radiografia influyeron mas en la decision del
    modelo. Se hace todo en una sola pasada por el modelo, evitando
    predecir la imagen dos veces.

    Args:
        array: arreglo NumPy de la imagen original (sin preprocesar),
            tal como la devuelven read_dicom_file o read_jpg_file.

    Returns:
        Una tupla (label, proba, heatmap):
        - label: la clase predicha ("bacteriana", "normal" o "viral").
        - proba: la probabilidad de esa clase, en porcentaje (0-100).
        - heatmap: arreglo NumPy de la imagen original con el mapa de
          calor superpuesto en colores.
    """
    img = preprocess(array)
    model = model_fun()
    last_conv_layer = model.get_layer("conv10_thisone")
    grad_model = tf.keras.models.Model(
        inputs=model.inputs, outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_layer_output, preds = grad_model([img])
        if isinstance(preds, list):
            preds = preds[0]
        argmax = tf.argmax(preds[0])
        output = preds[:, argmax]

    # Se extraen la clase y la probabilidad a partir de la misma
    # prediccion ya calculada arriba, sin volver a correr el modelo.
    prediction = int(argmax.numpy())
    proba = float(tf.reduce_max(preds).numpy()) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"

    grads = tape.gradient(output, conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_layer_output_value = conv_layer_output[0].numpy()
    pooled_grads_value = pooled_grads.numpy()
    for filters in range(64):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)

    return label, proba, superimposed_img[:, :, ::-1]