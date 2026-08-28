"""Modulo encargado de cargar el modelo entrenado para deteccion de neumonia."""

import os
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

MODEL_FILENAME = os.environ.get("MODEL_PATH", "conv_MLP_84.h5")

_model_cache = None


def model_fun():
    """Carga el modelo entrenado y lo retorna listo para hacer predicciones.

    El nombre del archivo del modelo se toma de la variable de entorno
    MODEL_PATH si esta definida; si no, usa "conv_MLP_84.h5" por
    defecto. Esto permite apuntar a un modelo con otro nombre 
    sin modificar el codigo.

    El modelo se carga la primera vez que se llama esta funcion, y se
    guarda en cache (_model_cache) para que las siguientes llamadas no
    tengan que volver a leerlo del disco.

    Se usa compile=False porque el archivo .h5 fue guardado con una
    version anterior de Keras, y su configuracion de entrenamiento
    (optimizador y funcion de perdida) ya no es compatible con Keras 3.
    Como el modelo solo se usa para predecir, no para seguir
    entrenandolo, esa configuracion no se necesita.

    Returns:
        El modelo de Keras ya cargado, listo para llamar a .predict().

    Raises:
        FileNotFoundError: si el archivo del modelo (MODEL_FILENAME) no
            se encuentra en la carpeta raiz del proyecto.
    """
    global _model_cache
    if _model_cache is None:
        if not os.path.exists(MODEL_FILENAME):
            raise FileNotFoundError(
                f"No se encontro {MODEL_FILENAME}. Este archivo no esta en "
                "el repositorio (ver .gitignore); solicitalo al equipo, o "
                "define la variable de entorno MODEL_PATH apuntando a tu "
                "archivo de modelo."
            )
        _model_cache = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
    return _model_cache