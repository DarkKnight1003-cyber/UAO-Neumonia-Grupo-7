import pytest

import load_model


@pytest.mark.parametrize("present", [False, True] * 10)
def test_model_fun_handles_missing_file_and_caches_loaded_model(monkeypatch, present):
    load_model._model_cache = None
    loaded = object()
    load_calls = []
    monkeypatch.setattr(load_model.os.path, "exists", lambda _: present)
    monkeypatch.setattr(load_model.tf.keras.models, "load_model", lambda path, compile: load_calls.append((path, compile)) or loaded)
    if not present:
        with pytest.raises(FileNotFoundError, match="conv_MLP_84.h5"):
            load_model.model_fun()
        assert load_calls == []
    else:
        assert load_model.model_fun() is loaded
        assert load_model.model_fun() is loaded
        assert load_calls == [("conv_MLP_84.h5", False)]