"""Pruebas unitarias para el modulo load_model (carga del modelo entrenado)."""

import pytest

import load_model


@pytest.mark.parametrize("present", [False, True] * 10)
def test_model_fun_handles_missing_file_and_caches_loaded_model(monkeypatch, present):
    """Verifica el manejo de archivo faltante y el cacheo del modelo cargado.

    Cuando el archivo del modelo no existe, model_fun debe lanzar
    FileNotFoundError sin llegar a intentar cargarlo. Cuando si existe,
    debe cargarlo una sola vez y reutilizar esa misma instancia en
    llamadas posteriores, gracias al cache interno (_model_cache).

    Args:
        monkeypatch: fixture de pytest para simular os.path.exists y
            tf.keras.models.load_model.
        present: indica si el archivo del modelo existe o no
            (parametrizado).
    """
    # Arrange
    load_model._model_cache = None
    loaded = object()
    load_calls = []
    monkeypatch.setattr(load_model.os.path, "exists", lambda _: present)
    monkeypatch.setattr(load_model.tf.keras.models, "load_model", lambda path, compile: load_calls.append((path, compile)) or loaded)

    # Act & Assert
    if not present:
        with pytest.raises(FileNotFoundError, match="conv_MLP_84.h5"):
            load_model.model_fun()
        assert load_calls == []
    else:
        assert load_model.model_fun() is loaded
        assert load_model.model_fun() is loaded
        assert load_calls == [("conv_MLP_84.h5", False)]