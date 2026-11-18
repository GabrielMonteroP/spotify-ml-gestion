import pandas as pd
import numpy as np
import os
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def entrenar_y_guardar_modelo(ruta_datos_limpios, ruta_modelo_salida):
    print("=== INICIANDO FASE DE ENTRENAMIENTO ===")
    
    print(f"Cargando datos limpios desde {ruta_datos_limpios}...")
    df = pd.read_csv(ruta_datos_limpios)
    
    # 1. Clave de agrupación exacta de tu notebook (evita fugas por mayúsculas/espacios)
    df["grupo_cancion"] = (df["artists"].str.lower().str.strip() + "|" + 
                           df["track_name"].str.lower().str.strip())
    
    # 2. Definir las columnas numéricas
    cols_num = ["danceability", "energy", "loudness", "speechiness", "acousticness",
                "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature"]
    
    # XGBoost prefiere booleanos como 0 y 1
    if 'explicit' in df.columns:
        df['explicit'] = df['explicit'].astype(int)
        
    y = df["popularity"]
    grupos = df["grupo_cancion"]
    
    print("Separando datos con GroupShuffleSplit...")
    # 3. División agrupada por canción (Lógica exacta de la EP3)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_train, idx_test = next(gss.split(df, y, groups=grupos))
    
    X_train_full = df.iloc[idx_train].copy()
    X_test_full = df.iloc[idx_test].copy()
    y_train = y.iloc[idx_train].copy()
    y_test = y.iloc[idx_test].copy()
    
    # Verificación de fuga por consola
    solap_id = set(X_train_full["track_id"]) & set(X_test_full["track_id"])
    solap_grupo = set(X_train_full["grupo_cancion"]) & set(X_test_full["grupo_cancion"])
    print(f"Train: {len(X_train_full)} | Test: {len(X_test_full)}")
    print(f"Fuga -> track_id solapados: {len(solap_id)} | canciones solapadas: {len(solap_grupo)}")
    
    # 4. Filtrar solo las columnas que el modelo necesita (botamos IDs, grupos, etc.)
    columnas_modelo = cols_num + ["explicit", "track_genre"]
    X_train = X_train_full[columnas_modelo]
    X_test = X_test_full[columnas_modelo]
    
    print("Configurando el Pipeline Maestro...")
    # 5. Preprocesamiento: Imputar, escalar y One-Hot
    pipeline_numerico = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler())
    ])
    
    preproceso = ColumnTransformer(
        transformers=[
            ("num", pipeline_numerico, cols_num),
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), ["track_genre"])
        ],
        remainder="passthrough"
    )
    
    # 6. Pipeline final
    pipeline_final = Pipeline(steps=[
        ("preprocesamiento", preproceso),
        ("modelo", XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42))
    ])
    
    print("Entrenando el Pipeline...")
    pipeline_final.fit(X_train, y_train)
    
    # 7. Evaluar
    y_pred = pipeline_final.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Métricas de Validación -> MAE: {mae:.3f} | R2: {r2:.3f}")
    
    # 8. Guardar
    print(f"Exportando Pipeline Maestro a {ruta_modelo_salida}...")
    os.makedirs(os.path.dirname(ruta_modelo_salida), exist_ok=True)
    joblib.dump(pipeline_final, ruta_modelo_salida)
    
    print("=== ENTRENAMIENTO COMPLETADO ===")

if __name__ == "__main__":
    RUTA_IN = "data/spotify_limpio.csv"
    RUTA_OUT = "models/pipeline_xgboost.joblib"
    
    entrenar_y_guardar_modelo(RUTA_IN, RUTA_OUT)
    