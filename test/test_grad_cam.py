import numpy as np
import pytest

import grad_cam


@pytest.mark.parametrize("error", [ValueError, TypeError, RuntimeError, KeyError, OSError] * 4)
def test_grad_cam_propagates_preprocess_errors(monkeypatch, error):
    def fail(_):
        raise error("invalid image")

    monkeypatch.setattr(grad_cam, "preprocess", fail)
    with pytest.raises(error, match="invalid image"):
        grad_cam.grad_cam(np.ones((2, 2, 3), dtype=np.uint8))