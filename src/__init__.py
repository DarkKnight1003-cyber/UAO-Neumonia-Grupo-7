"""Paquete src: configura TensorFlow antes de que cualquier modulo lo importe."""

import os

# Se reduce el nivel de logging de TensorFlow para que no imprima sus
# mensajes informativos de inicio (oneDNN, instrucciones de CPU,
# disponibilidad de GPU). Esto debe hacerse ANTES de que TensorFlow se
# importe en cualquier parte del programa, por eso va aqui: __init__.py
# se ejecuta automaticamente la primera vez que se importa algo de src.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")