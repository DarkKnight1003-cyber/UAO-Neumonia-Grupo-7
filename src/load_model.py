"""Modulo encargado de cargar el modelo entrenado para deteccion de neumonia."""

import os
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
_model_cache = None


def model_fun():
    """Carga el modelo entrenado y lo retorna listo para hacer predicciones.

    El modelo se carga desde el archivo conv_MLP_84.h5 la primera vez que
    se llama esta funcion, y se guarda en cache (_model_cache) para que
    las siguientes llamadas no tengan que volver a leerlo del disco,
    haciendo que las predicciones posteriores sean mucho mas rapidas.

    Se usa compile=False porque el archivo .h5 fue guardado con una
    version anterior de Keras, y su configuracion de entrenamiento
    (optimizador y funcion de perdida) ya no es compatible con Keras 3.
    Como el modelo solo se usa para predecir, no para seguir
    entrenandolo, esa configuracion no se necesita.

    Returns:
        El modelo de Keras ya cargado, listo para llamar a .predict().

    Raises:
        FileNotFoundError: si el archivo conv_MLP_84.h5 no se encuentra
            en la carpeta raiz del proyecto.
    """
    global _model_cache
    if _model_cache is None:
        if not os.path.exists("conv_MLP_84.h5"):
            raise FileNotFoundError(
                "No se encontro conv_MLP_84.h5. Este archivo no esta en el "
                "repositorio (ver .gitignore); solicitalo al equipo."
            )
        _model_cache = tf.keras.models.load_model("conv_MLP_84.h5", compile=False)
    return _model_cache