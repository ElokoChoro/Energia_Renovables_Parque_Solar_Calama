import requests
import json

def get_solar_data(latitude=-22.4254555, longitude=-68.8597502):
    """
    Fetches solar radiation and weather data for a given location (default is Parque Solar Calama III) 
    from the Open-Meteo API.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    # Parameters specifically useful for a solar park
    params = {
        "latitude": latitude, 
        "longitude": longitude,
        "hourly": ["temperature_2m", "direct_radiation", "diffuse_radiation", "direct_normal_irradiance", "cloudcover"],
        "timezone": "America/Santiago"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Raise an exception for bad status codes
        
        data = response.json()
        print("Datos obtenidos exitosamente desde Open-Meteo API.")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Error al obtener los datos: {e}")
        return None

if __name__ == "__main__":
    print("Iniciando consulta a la API de Open-Meteo para Parque Solar Calama III...")
    data = get_solar_data()
    
    if data:
        # Muestra una muestra de los datos (la primera hora)
        print("\nMuestra de datos de la primera hora disponible:")
        print(f"Hora: {data['hourly']['time'][0]}")
        print(f"Temperatura: {data['hourly']['temperature_2m'][0]} °C")
        print(f"Radiación Directa Normal (DNI): {data['hourly']['direct_normal_irradiance'][0]} W/m²")
        print(f"Cobertura de nubes: {data['hourly']['cloudcover'][0]} %")
        
        # Opcional: Guardar los datos en un archivo JSON local
        with open('datos_calama_meteo.json', 'w') as f:
            json.dump(data, f, indent=4)
        print("\nTodos los datos han sido guardados en 'datos_calama_meteo.json'")
