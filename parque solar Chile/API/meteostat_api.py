import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# ========================================================
# Configuración de la API: Meteostat a través de RapidAPI
# ========================================================
API_KEY = "7aa47fdd29msha79c80b50a60f46p1fc0a2jsnaf40c865bebc"
API_HOST = "meteostat.p.rapidapi.com"
URL_BASE = "https://meteostat.p.rapidapi.com/stations/hourly"

# ID de la estación meteorológica de Calama (WMO ID: 85432)
ESTACION_CALAMA = "85432"

# Coordenadas exactas para otros cálculos (modelo astronómico)
LATITUD_PARQUE  = -22.4254555
LONGITUD_PARQUE = -68.8597502

def get_solar_data(dias_historial=365):
    """
    Descarga datos horarios reales de la estación Calama (85432)
    usando la REST API de Meteostat (vía RapidAPI).
    
    Retorna:
    --------
    DataFrame de pandas con índice de fecha y variables climáticas,
    o None en caso de error.
    """
    fecha_fin = datetime.now() - timedelta(days=1)
    fecha_inicio = fecha_fin - timedelta(days=dias_historial)
    
    start_str = fecha_inicio.strftime('%Y-%m-%d')
    end_str = fecha_fin.strftime('%Y-%m-%d')
    
    print(f"  Fuente de datos: Meteostat REST API (via RapidAPI)")
    print(f"  Estacion: Calama (ID: {ESTACION_CALAMA})")
    print(f"  Rango solicitado: {start_str} -> {end_str}")
    print("  Conectando con la API y descargando (puede tomar unos segundos)...\n")

    headers = {
        "x-rapidapi-host": API_HOST,
        "x-rapidapi-key": API_KEY
    }

    all_data = []
    
    
    current_start = fecha_inicio
    
    try:
        while current_start < fecha_fin:
            current_end = min(current_start + timedelta(days=29), fecha_fin)
            
            querystring = {
                "station": ESTACION_CALAMA,
                "start": current_start.strftime('%Y-%m-%d'),
                "end": current_end.strftime('%Y-%m-%d')
            }
            
            response = requests.get(URL_BASE, headers=headers, params=querystring, timeout=30)
            response.raise_for_status()
            
            datos_json = response.json()
            if "data" in datos_json and len(datos_json["data"]) > 0:
                all_data.extend(datos_json["data"])
                
            current_start = current_end + timedelta(days=1)
            
        if not all_data:
            print("  [ERROR] La API no devolvio datos para este rango de fechas.")
            return None
            
        # Convertir JSON a DataFrame
        df = pd.DataFrame(all_data)
        
        # Convertir la columna 'time' a datetime real
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")
        
        n_registros = len(df)
        cobertura = round(n_registros / (dias_historial * 24) * 100, 1)
        
        print(f"  [OK] Datos reales descargados: {n_registros} registros horarios")
        print(f"  [OK] Cobertura temporal: {cobertura}% del periodo solicitado")
        print(f"  [OK] Periodo real: {df.index.min()} -> {df.index.max()}\n")

        return df

    except requests.exceptions.RequestException as e:
        print(f"  Error de conexión a la API: {e}")
        return None
    except Exception as e:
        print(f"  Error procesando datos: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("   PRUEBA DE CONEXIÓN: METEOSTAT RAPIDAPI -> CALAMA III")
    print("=" * 60 + "\n")

    df_prueba = get_solar_data(dias_historial=30)

    if df_prueba is not None:
        print("Primeras 5 filas:")
        print(df_prueba.head())
        print("\nColumnas disponibles:")
        print(df_prueba.columns.tolist())
