import importlib
import uvicorn

# Importa src/03_api.py respetando la numeración modular
modulo_api = importlib.import_module("src.03_api")
app = modulo_api.app

if __name__ == "__main__":
    print("=== LEVANTANDO SERVIDOR FASTAPI ===")
    print("Documentación interactiva (Swagger): http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)