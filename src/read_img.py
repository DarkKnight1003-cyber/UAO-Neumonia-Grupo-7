"""Modulo encargado de leer archivos de imagenes radiograficas (DICOM y JPG)."""

import cv2
import numpy as np
import pydicom as dicom
from PIL import Image


def read_dicom_file(path):
    """Lee un archivo DICOM y lo retorna listo para procesar y mostrar.

    Se extrae la matriz de pixeles del archivo DICOM, se normalizan sus
    valores entre 0 y 255, y se convierte de escala de grises a formato
    RGB (3 canales), ya que el resto del pipeline espera ese formato.

    Args:
        path: ruta del archivo .dcm a leer.

    Returns:
        Una tupla (img_RGB, img2show):
        - img_RGB: arreglo NumPy en formato RGB, listo para el preprocesamiento.
        - img2show: la misma imagen en formato PIL, lista para mostrarse
          en la interfaz grafica.
    """
    img = dicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    img_RGB = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_RGB, img2show


def read_jpg_file(path):
    """Lee un archivo JPG/JPEG y lo retorna listo para procesar y mostrar.

    A diferencia del DICOM, un JPG ya viene en un formato de imagen
    estandar, asi que solo se normalizan sus valores de pixel entre
    0 y 255 para dejarlo consistente con el resultado de read_dicom_file.

    Args:
        path: ruta del archivo .jpg/.jpeg a leer.

    Returns:
        Una tupla (img2, img2show):
        - img2: arreglo NumPy listo para el preprocesamiento.
        - img2show: la misma imagen en formato PIL, lista para mostrarse
          en la interfaz grafica.
    """
    img = cv2.imread(path)
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show