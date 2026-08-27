import numpy as np
import pytest

import preprocess_img


@pytest.mark.parametrize("shape", [(1, 1, 3), (2, 3, 3), (4, 5, 3), (8, 2, 3), (10, 10, 3)] * 4)
def test_preprocess_returns_normalized_batch_tensor(shape):
    source = np.full(shape, 127, dtype=np.uint8)
    result = preprocess_img.preprocess(source)
    assert result.shape == (1, 512, 512, 1)
    assert result.dtype.kind == "f"
    assert 0 <= result.min() <= result.max() <= 1