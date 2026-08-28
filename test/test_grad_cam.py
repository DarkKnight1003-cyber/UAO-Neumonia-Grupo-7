"""Pruebas unitarias para el modulo grad_cam (clasificacion y mapa de calor)."""

import numpy as np
import pytest

import grad_cam


@pytest.mark.parametrize("error", [ValueError, TypeError, RuntimeError, KeyError, OSError] * 4)
def test_grad_cam_propagates_preprocess_errors(monkeypatch, error):
    """Verifica que grad_cam propague cualquier error lanzado por preprocess.

    Simula preprocess para que falle con distintos tipos de excepcion,
    y confirma que grad_cam no las oculte ni las transforme, sino que
    las deje propagar tal cual hacia quien la llamo.

    Args:
        monkeypatch: fixture de pytest para simular grad_cam.preprocess.
        error: clase de excepcion a simular (parametrizado).
    """
    # Arrange
    def fail(_):
        raise error("invalid image")

    monkeypatch.setattr(grad_cam, "preprocess", fail)

    # Act & Assert
    with pytest.raises(error, match="invalid image"):
        grad_cam.grad_cam(np.ones((2, 2, 3), dtype=np.uint8))