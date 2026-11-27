# Dockerfile de la API Flask del Proyecto Final (Casa Aguacate) para correr en la nube.
# Sirve para: construir la imagen que ejecuta gunicorn + Flask + TensorFlow (CPU) en Hugging Face Spaces
# (SDK Docker, puerto 7860, usuario 1000) o en cualquier proveedor que ejecute contenedores.
# El contexto de construccion lo arma preparar_contexto.sh (carpeta con casaguacate/, artefactos/, servidor.py,
# requirements.txt, arrancar_nube.sh y chatbot_hospedaje.py). API_TOKEN se inyecta como secreto del proveedor,
# nunca va en la imagen.
FROM python:3.11-slim

# Usuario sin privilegios con UID 1000 (lo pide Hugging Face Spaces).
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860
# La API espera la forma del repo: <raiz>/chatbot_hospedaje.py y <raiz>/proyecto_final/app/ (config.REPO_ROOT).
RUN mkdir -p /home/user/repo/proyecto_final/app && chown -R user:user /home/user/repo
WORKDIR /home/user/repo/proyecto_final/app

# Dependencias primero para aprovechar la cache de capas.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Chatbot del Avance 1 (base de conocimiento y textos de respuesta) en la raiz del repo.
COPY --chown=user chatbot_hospedaje.py /home/user/repo/chatbot_hospedaje.py
# Codigo de la API y modelos entrenados (TF-IDF+SVM y cabeza de vision).
COPY --chown=user casaguacate ./casaguacate
COPY --chown=user artefactos ./artefactos
COPY --chown=user servidor.py arrancar_nube.sh ./
RUN chmod +x arrancar_nube.sh

USER user

# Descarga los pesos ImageNet de EfficientNetB0 durante la construccion para que el arranque no dependa de internet.
RUN python -c "from casaguacate.vision import build_extractor; build_extractor('efficientnetb0')"

EXPOSE 7860
CMD ["./arrancar_nube.sh"]
