"""Publica la API de Cas'aguacate en un Hugging Face Space (SDK Docker) con huggingface_hub.

Sirve para: crear el Space si no existe, guardar el secreto API_TOKEN y subir la carpeta armada por
preparar_contexto.sh (maneja bien los .joblib binarios). Lo llama publicar_space.sh; no se corre a mano.

Autenticacion: HfApi() usa la sesion guardada por `hf auth login` o la variable HF_TOKEN. El token de
Hugging Face nunca se pide, no va en la URL ni en argumentos. El valor de API_TOKEN llega por la variable
de entorno TOKEN_VAL (no por argumentos) y nunca se imprime.
Uso interno: publicar_space.py <carpeta-contexto> <nombre-del-space>
"""
import os
import sys

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError


def fallar(mensaje: str) -> None:
    """Termina con un mensaje claro (codigo 1) en vez de un traceback largo."""
    print(f"ERROR: {mensaje}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """Crea/actualiza el Space, sube el secreto y el codigo, e imprime la URL publica."""
    if len(sys.argv) != 3:
        fallar("uso: publicar_space.py <carpeta-contexto> <nombre-del-space>")
    carpeta, nombre = sys.argv[1], sys.argv[2]
    secreto = os.environ.get("TOKEN_VAL", "")
    if len(secreto) < 32:
        fallar("falta TOKEN_VAL (API_TOKEN) o es demasiado corto.")

    api = HfApi()
    try:
        usuario = api.whoami()["name"]
    except Exception as exc:  # sin sesion, token invalido o sin red
        fallar("no hay sesion valida de Hugging Face. Corre `hf auth login` (o define HF_TOKEN). "
               f"Detalle: {type(exc).__name__}")
    repo = f"{usuario}/{nombre}"
    try:
        api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
        api.add_space_secret(repo, "API_TOKEN", secreto)
        api.upload_folder(folder_path=carpeta, repo_id=repo, repo_type="space",
                          commit_message="Despliegue de la API de Cas'aguacate")
    except HfHubHTTPError as exc:
        codigo = exc.response.status_code if exc.response is not None else "?"
        pista = " (el token necesita permiso de escritura)" if codigo in (401, 403) else ""
        fallar(f"Hugging Face respondio HTTP {codigo}{pista} al publicar {repo}.")
    subdominio = f"{usuario}-{nombre}".lower().replace("_", "-").replace(".", "-")
    print(f"Space: https://huggingface.co/spaces/{repo}")
    print(f"URL publica de la API: https://{subdominio}.hf.space (la construccion tarda unos minutos)")


if __name__ == "__main__":
    main()
