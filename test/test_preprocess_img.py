"""Pruebas unitarias para el modulo preprocess_img (preprocesamiento de imagenes)."""

import numpy as np
import pytest

import preprocess_img


@pytest.mark.parametrize("shape", [(1, 1, 3), (2, 3, 3), (4, 5, 3), (8, 2, 3), (10, 10, 3)] * 4)
def test_preprocess_returns_normalized_batch_tensor(shape):
    """Verifica que preprocess devuelva un tensor normalizado con forma (1, 512, 512, 1).

    Args:
        shape: forma del arreglo de entrada de prueba (parametrizada).
    """
    # Arrange
    source = np.full(shape, 127, dtype=np.uint8)

    # Act
    result = preprocess_img.preprocess(source)

    # Assert
    assert result.shape == (1, 512, 512, 1)
    assert result.dtype.kind == "f"
    assert 0 <= result.min() <= result.max() <= 1