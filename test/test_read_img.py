"""Pruebas unitarias para el modulo read_img (lectura de imagenes DICOM y JPG)."""

import numpy as np
import pytest
from PIL import Image

import read_img


@pytest.mark.parametrize("shape", [(3, 4, 3), (4, 5, 3), (8, 2, 3), (1, 7, 3), (10, 10, 3), (2, 9, 3), (7, 3, 3), (5, 6, 3), (9, 1, 3), (12, 4, 3)])
def test_read_jpg_file_normalizes_and_preserves_shape(monkeypatch, shape):
    """Verifica que read_jpg_file normalice los valores de pixel y conserve la forma de la imagen.

    Simula cv2.imread con un arreglo de prueba de una forma dada, y
    confirma que read_jpg_file devuelva un arreglo normalizado entre
    0 y 255, con la misma forma que la imagen original, y una imagen
    PIL valida para mostrar en la interfaz.

    Args:
        monkeypatch: fixture de pytest para simular cv2.imread.
        shape: forma del arreglo de prueba (parametrizada).
    """
    source = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape) + 1
    monkeypatch.setattr(read_img.cv2, "imread", lambda _: source)
    result, shown = read_img.read_jpg_file("image.jpg")
    expected = np.uint8(source.astype(float) / source.max() * 255.0)
    assert result.shape == shape
    assert result.dtype == np.uint8
    assert np.array_equal(result, expected)
    assert isinstance(shown, Image.Image)


@pytest.mark.parametrize("shape", [(2, 3), (4, 4), (5, 2), (1, 8), (6, 7), (3, 6), (7, 2), (8, 8), (2, 10), (9, 5)])
def test_read_dicom_file_converts_grayscale_to_rgb(monkeypatch, shape):
    """Verifica que read_dicom_file convierta correctamente de escala de grises a RGB.

    Simula dicom.dcmread con una matriz de pixeles de prueba, y
    confirma que read_dicom_file devuelva un arreglo RGB (3 canales
    iguales, ya que la imagen original es en escala de grises) con
    los valores normalizados entre 0 y 255.

    Args:
        monkeypatch: fixture de pytest para simular dicom.dcmread.
        shape: forma de la matriz de pixeles de prueba (parametrizada).
    """
    pixels = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape) + 1
    monkeypatch.setattr(read_img.dicom, "dcmread", lambda _: type("Dicom", (), {"pixel_array": pixels})())
    result, shown = read_img.read_dicom_file("image.dcm")
    expected = np.uint8(pixels / pixels.max() * 255)
    assert result.shape == (*shape, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(result[:, :, 0], expected)
    assert np.array_equal(result[:, :, 1], expected)
    assert isinstance(shown, Image.Image)