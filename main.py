import threading
import webbrowser
from pathlib import Path
import uvicorn


def abrir_frontend():
    ruta_html = Path("frontend/index.html").resolve().as_uri()
    webbrowser.open(ruta_html)


if __name__ == "__main__":
    print("=== LEVANTANDO SERVIDOR FASTAPI ===")
    print("Documentación interactiva (Swagger): http://127.0.0.1:8000/docs")
    print("Abriendo dashboard en el navegador...")

    threading.Timer(3.5, abrir_frontend).start()

    uvicorn.run("src.03_api:app", host="127.0.0.1", port=8000)