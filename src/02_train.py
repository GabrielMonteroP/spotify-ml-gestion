import pandas as pd
import numpy as np
import os
import json
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def entrenar_y_guardar_modelo(ruta_datos_limpios, ruta_modelo_salida, ruta_metricas_salida):
    print("=== INICIANDO FASE DE ENTRENAMIENTO ===")
    
    print(f"Cargando datos limpios desde {ruta_datos_limpios}...")
    df = pd.read_csv(ruta_datos_limpios)
    
    # 1. Clave de agrupación exacta de tu notebook (evita fugas por mayúsculas/espacios)
    df["grupo_cancion"] = (df["artists"].str.lower().str.strip() + "|" + 
                           df["track_name"].str.lower().str.strip())
    
    # 1.5 Etiqueta auxiliar de tipo de artista (NO entra al modelo, solo para KPIs de equidad)
    UMBRAL_CONSOLIDADO = 10
    artista_principal = df["artists"].str.split(";").str[0].str.strip()
    n_canciones_artista = artista_principal.value_counts()
    df["tipo_artista"] = np.where(
        artista_principal.map(n_canciones_artista) >= UMBRAL_CONSOLIDADO,
        "Consolidado", "Emergente"
    )
    
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
    
    # 4. Filtrar solo las columnas que el modelo necesita (botamos IDs, grupos, tipo_artista, etc.)
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
    
    # 7. Evaluación Global y por Subgrupos
    y_pred = pipeline_final.predict(X_test)
    mae_global = mean_absolute_error(y_test, y_pred)
    r2_global = r2_score(y_test, y_pred)
    print(f"\n=== MÉTRICAS GLOBALES ===")
    print(f"MAE Global: {mae_global:.3f} | R2 Global: {r2_global:.3f}")
    
    # DataFrame auxiliar para calcular errores por subgrupo
    df_eval = pd.DataFrame({
        "y_real": y_test.values,
        "y_pred": y_pred,
        "error_abs": np.abs(y_test.values - y_pred),
        "track_genre": X_test_full["track_genre"].values,
        "tipo_artista": X_test_full["tipo_artista"].values
    })
    
    # 7.1 MAE por tipo de artista
    mae_artista = df_eval.groupby("tipo_artista")["error_abs"].mean().round(3)
    brecha_artista = round(float(mae_artista.max() - mae_artista.min()), 3)
    print("\n=== MAE POR TIPO DE ARTISTA ===")
    for tipo, valor in mae_artista.items():
        print(f"  - {tipo}: {valor:.3f}")
    print(f"  -> Brecha (Consolidado vs Emergente): {brecha_artista:.3f}")
    
    # 7.2 MAE por género musical
    mae_genero = df_eval.groupby("track_genre")["error_abs"].mean().sort_values().round(3)
    mejor_genero = mae_genero.index[0]
    peor_genero = mae_genero.index[-1]
    brecha_genero = round(float(mae_genero.iloc[-1] - mae_genero.iloc[0]), 3)
    
    print("\n=== MAE POR GÉNERO (TOP 5 MEJORES Y PEORES) ===")
    print(f"  Mejor género: {mejor_genero} (MAE: {mae_genero.iloc[0]:.3f})")
    print(f"  Peor género:  {peor_genero} (MAE: {mae_genero.iloc[-1]:.3f})")
    print(f"  -> Brecha entre géneros: {brecha_genero:.3f}")
    print("\n  Top 5 menor error:")
    print(mae_genero.head(5).to_string())
    print("\n  Top 5 mayor error:")
    print(mae_genero.tail(5).to_string())
    
    # 8. Guardar modelo y métricas JSON para el dashboard
    os.makedirs(os.path.dirname(ruta_modelo_salida), exist_ok=True)
    
    print(f"\nExportando Pipeline Maestro a {ruta_modelo_salida}...")
    joblib.dump(pipeline_final, ruta_modelo_salida)
    
    metricas_json = {
        "mae_global": round(float(mae_global), 3),
        "r2_global": round(float(r2_global), 3),
        "por_tipo_artista": {
            "mae": {k: float(v) for k, v in mae_artista.items()},
            "brecha": brecha_artista
        },
        "por_genero": {
            "mejor_genero": {"nombre": mejor_genero, "mae": float(mae_genero.iloc[0])},
            "peor_genero": {"nombre": peor_genero, "mae": float(mae_genero.iloc[-1])},
            "brecha": brecha_genero,
            "todos": {k: float(v) for k, v in mae_genero.items()}
        }
    }
    
    with open(ruta_metricas_salida, "w", encoding="utf-8") as f:
        json.dump(metricas_json, f, indent=4, ensure_ascii=False)
    print(f"Métricas de equidad exportadas a {ruta_metricas_salida}")
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")

if __name__ == "__main__":
    RUTA_IN = "data/spotify_limpio.csv"
    RUTA_OUT = "models/pipeline_xgboost.joblib"
    RUTA_METRICAS = "models/metricas_modelo.json"
    
    entrenar_y_guardar_modelo(RUTA_IN, RUTA_OUT, RUTA_METRICAS)