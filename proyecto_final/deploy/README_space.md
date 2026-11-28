---
title: "Cas'aguacate API"
emoji: 🏡
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Cas'aguacate API

API REST (Flask) del Proyecto Final de la maestria en IA (Tecmilenio): chatbot NLP (TF-IDF + SVM por idioma),
clasificador de fotos (EfficientNetB0 + cabeza) y agente. Endpoints: `/health`, `/chat`, `/image`, `/agent`, `/metrics`.
Todos menos `/health` exigen el encabezado `X-API-Token` (secreto `API_TOKEN` del Space). Demo academica.

Este archivo es la cabecera del Space; `publicar_space.sh` lo copia como `README.md` del Space.
