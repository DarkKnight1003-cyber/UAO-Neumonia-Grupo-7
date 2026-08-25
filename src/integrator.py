"""Modulo encargado de orquestar el pipeline completo de prediccion."""

from src.grad_cam import grad_cam


def predict(array):
    """Ejecuta el pipeline completo de deteccion sobre una imagen.

    Punto de entrada principal del sistema: recibe la imagen ya leida
    (por read_dicom_file o read_jpg_file) y delega en grad_cam el
    trabajo de preprocesarla, clasificarla y generar su mapa de calor.

    Args:
        array: arreglo NumPy de la imagen original, tal como la
            devuelven read_dicom_file o read_jpg_file.

    Returns:
        Una tupla (label, proba, heatmap), tal como la retorna
        grad_cam: la clase predicha, su probabilidad y la imagen
        con el mapa de calor superpuesto.
    """
    label, proba, heatmap = grad_cam(array)
    return label, proba, heatmap