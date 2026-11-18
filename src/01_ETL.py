import pandas as pd
import numpy as np
import os

def cargar_datos(ruta_archivo):
    """Carga el dataset de Spotify desde el archivo CSV original."""
    print(f"Cargando datos desde: {ruta_archivo}")
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"No se encontró el archivo en {ruta_archivo}")
    
    # Se asume que la columna 0 es el índice (como en tu notebook)
    return pd.read_csv(ruta_archivo, index_col=0)

def limpiar_datos(df):
    """Ejecuta imputación, elimina duplicados y estandariza los datos."""
    print("Iniciando limpieza y transformación de datos...")
    
    # 1. Eliminar nulos en columnas críticas y duplicados exactos
    df_limpio = df.dropna(subset=["artists", "album_name", "track_name"]).drop_duplicates().copy()
    
    # 2. Convertir valores físicamente imposibles a NaN (para imputarlos luego en el Pipeline)
    df_limpio.loc[df_limpio["tempo"] == 0, "tempo"] = np.nan
    df_limpio.loc[df_limpio["time_signature"] < 3, "time_signature"] = np.nan
    
    # 3. Separar las canciones con popularidad 0 (datos faltantes según EDA)
    df_validos = df_limpio[df_limpio["popularity"] > 0].copy()
    df_ceros = df_limpio[df_limpio["popularity"] == 0].copy()
    
    print(f"Limpieza finalizada. Registros válidos: {len(df_validos)} | Registros pop=0: {len(df_ceros)}")
    return df_validos, df_ceros

def exportar_datos(df, ruta_salida):
    """Exporta el dataset completamente limpio para la siguiente fase."""
    print(f"Exportando datos a: {ruta_salida}")
    os.makedirs(os.path.dirname(ruta_salida), exist_ok=True)
    df.to_csv(ruta_salida, index=False)

def ejecutar_pipeline_etl(ruta_entrada, ruta_salida_limpio, ruta_salida_ceros=None):
    """Orquesta todo el proceso ETL conectando las funciones anteriores."""
    print("=== INICIANDO PIPELINE ETL ===")
    
    # Extract
    df_crudo = cargar_datos(ruta_entrada)
    
    # Transform
    df_limpio, df_ceros = limpiar_datos(df_crudo)
    
    # Load
    exportar_datos(df_limpio, ruta_salida_limpio)
    if ruta_salida_ceros:
        exportar_datos(df_ceros, ruta_salida_ceros)
        
    print("=== PIPELINE ETL COMPLETADO ===")
    return df_limpio

# Bloque de ejecución principal (solo se ejecuta si corres este script directamente)
if __name__ == "__main__":
    # Rutas relativas asumiendo que ejecutas desde la raíz del proyecto
    RUTA_IN = "data/Spotify_Tracks_Dataset.csv"
    RUTA_OUT = "data/spotify_limpio.csv"
    RUTA_CEROS = "data/spotify_ceros.csv"
    
    ejecutar_pipeline_etl(RUTA_IN, RUTA_OUT, RUTA_CEROS)