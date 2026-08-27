import numpy as np
import pytest

import integrator


@pytest.mark.parametrize("value", [1, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 250, 100, 150, 200])
def test_integrator_delegates_and_returns_grad_cam_result(monkeypatch, value):
    expected = ("normal", float(value), np.array([value], dtype=np.uint8))
    calls = []

    def fake_grad_cam(array):
        calls.append(array)
        return expected

    monkeypatch.setattr(integrator, "grad_cam", fake_grad_cam)
    source = object()
    result = integrator.predict(source)
    assert result[:2] == expected[:2]
    assert np.array_equal(result[2], expected[2])
    assert calls == [source]