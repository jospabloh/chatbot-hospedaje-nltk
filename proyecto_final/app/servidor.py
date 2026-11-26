"""Entry point of the REST API.

Development:  python servidor.py                     (http://127.0.0.1:5000)
Production:   gunicorn -w 1 --threads 4 -b 127.0.0.1:5000 servidor:app
Use ONE worker process: metrics and the rate limiter live in memory (threads are fine).
"""
import os

from casaguacate.api import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "5000")), threaded=True)
