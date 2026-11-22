# Spotify Popularity ML

Proyecto de Machine Learning para predecir la popularidad (`0` a `100`) de canciones de Spotify a partir de sus características de audio y género musical, acompañado de una API REST en FastAPI y un dashboard local interactivo.

## Estructura del Proyecto

```text
03Gestion/
├── data/
│   ├── Spotify_Tracks_Dataset.csv   # Dataset original
│   ├── spotify_ceros.csv            # Registros filtrados con popularity == 0
│   └── spotify_limpio.csv           # Dataset limpio para entrenamiento
├── frontend/
│   └── index.html                   # Dashboard interactivo local (HTML/CSS/JS)
├── models/
│   ├── metricas_modelo.json         # Métricas globales y de equidad del modelo
│   └── pipeline_xgboost.joblib      # Pipeline de preprocesamiento + XGBoost
├── src/
│   ├── 01_ETL.py                    # Limpieza y separación de datos
│   ├── 02_train.py                  # Entrenamiento, evaluación y exportación
│   └── 03_api.py                    # API REST con FastAPI
├── .gitignore
├── main.py                          # Inicia la API y abre el dashboard
├── requirements.txt                 # Dependencias de Python
└── README.md
```

## Requisitos Previos

* **Python 3.10+** instalado.
* El dataset original ubicado en `data/Spotify_Tracks_Dataset.csv` (en caso de ejecutar el pipeline desde cero).

## Instalación

1. Clonar el repositorio y entrar a la carpeta del proyecto:
   ```bash
   git clone https://github.com/GabrielMonteroP/spotify-ml-gestion.git
   cd spotify-ml-gestion
   ```

2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Ejecución

> **Nota:** Todos los comandos deben ejecutarse desde la **raíz del proyecto** para que las rutas relativas (`data/`, `models/`, `frontend/`) funcionen correctamente.

### Opción A: Ejecución rápida (si el modelo ya está entrenado)

Si los archivos `models/pipeline_xgboost.joblib` y `models/metricas_modelo.json` ya existen en el repositorio, basta con ejecutar:

```bash
python main.py
```

Esto levantará el servidor en `http://127.0.0.1:8000` y abrirá automáticamente `frontend/index.html` en tu navegador tras 3.5 segundos.

### Opción B: Ejecución completa desde cero (ETL + Entrenamiento + App)

1. **Procesar y limpiar los datos (ETL):**
   ```bash
   python src/01_ETL.py
   ```
   Genera `data/spotify_limpio.csv` y `data/spotify_ceros.csv`.

2. **Entrenar el modelo y calcular métricas:**
   ```bash
   python src/02_train.py
   ```
   Genera `models/pipeline_xgboost.joblib` y `models/metricas_modelo.json`.

3. **Levantar la API y el dashboard:**
   ```bash
   python main.py
   ```

## Endpoints de la API

Con el servidor corriendo en `http://127.0.0.1:8000`:

* **Documentación interactiva (Swagger UI):** `http://127.0.0.1:8000/docs`
* **`GET /`**: Verifica que la API esté activa.
* **`GET /metricas`**: Retorna el contenido de `models/metricas_modelo.json` (MAE, R² y brechas de equidad).
* **`POST /predict`**: Recibe un JSON con las 13 características de la canción (`danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `time_signature`, `explicit`, `track_genre`) y devuelve la popularidad estimada (`0` a `100`) y si es hit potencial (`>= 70`).