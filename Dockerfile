FROM python:3.13-slim

WORKDIR /app

# Dependencias del sistema para Tkinter y OpenCV
RUN apt-get update && apt-get install -y \
    tk \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Instalar UV dentro de la imagen
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copiar archivos de dependencias
COPY pyproject.toml uv.lock README.md ./

# Instalar dependencias Python con UV
RUN uv sync --frozen --no-dev --no-install-project

# Copiar código fuente
COPY src ./src

# Crear archivo requerido por python-xlib
RUN touch /root/.Xauthority

# Usar el entorno virtual creado por UV
ENV PATH="/app/.venv/bin:$PATH"

# Servidor gráfico de Windows mediante VcXsrv
ENV DISPLAY=host.docker.internal:0.0

# Ejecutar la aplicación
CMD ["python", "-m", "src.detector_neumonia"]