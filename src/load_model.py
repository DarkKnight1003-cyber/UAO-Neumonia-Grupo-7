"""Modulo encargado de cargar el modelo entrenado para deteccion de neumonia."""

import os
import tensorflow as tf

_model_cache = None


def model_fun():
    """Carga el modelo (una sola vez) y lo retorna listo para inferencia."""
    global _model_cache
    if _model_cache is None:
        if not os.path.exists("conv_MLP_84.h5"):
            raise FileNotFoundError(
                "No se encontro conv_MLP_84.h5. Este archivo no esta en el "
                "repositorio (ver .gitignore); solicitalo al equipo."
            )
        _model_cache = tf.keras.models.load_model("conv_MLP_84.h5", compile=False)
    return _model_cache