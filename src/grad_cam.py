"""Modulo encargado de generar el mapa de calor Grad-CAM sobre la radiografia."""

import cv2
import numpy as np
import tensorflow as tf

from src.preprocess_img import preprocess
from src.load_model import model_fun


def grad_cam(array):
    """Genera una imagen con el mapa de calor Grad-CAM superpuesto."""
    img = preprocess(array)
    model = model_fun()
    last_conv_layer = model.get_layer("conv10_thisone")
    grad_model = tf.keras.models.Model(
        inputs=model.input, outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_layer_output, preds = grad_model(img)
        if isinstance(preds, list):
            preds = preds[0]
        argmax = tf.argmax(preds[0])
        output = preds[:, argmax]
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
    return superimposed_img[:, :, ::-1]