import os
import sys
import math
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# Importamos el módulo de la API Meteostat
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from API.meteostat_api import get_solar_data, LATITUD_PARQUE

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
plt.style.use("seaborn-v0_8-darkgrid")
COLORES = ["#E94560", "#0F3460", "#533483", "#16213E", "#1A1A2E", "#E94560"]
sns.set_palette(COLORES)

DIR_GRAFICOS = os.path.join(os.path.dirname(__file__), "..", "visualizaciones")
os.makedirs(DIR_GRAFICOS, exist_ok=True)

# Variables del parque solar
AREA_PANELES  = 500_000   # m² (superficie total de paneles)
EFICIENCIA    = 0.20      # 20% eficiencia base de paneles monocristalinos


# ============================================================
# 1. CÁLCULO DE RADIACIÓN SOLAR (modelo astronómico)
# ============================================================
def calcular_ghi_clearsky(timestamps, latitud):
    """
    Estima la irradiancia horizontal global (GHI) en condiciones de cielo
    despejado para cada timestamp, usando fórmulas astronómicas estándar.

    A 2377 m.s.n.m. en el Desierto de Atacama, la transmisividad atmosférica
    es muy alta (~0.78-0.82) debido a la escasa humedad y contaminación.

    El GHI real se obtiene multiplicando este valor por la fracción de sol
    (tsun/60) reportada por la estación meteorológica.
    """
    ghi_list = []
    for ts in timestamps:
        dia_anio    = ts.dayofyear
        hora        = ts.hour + ts.minute / 60.0

        # Declinación solar (δ) en grados
        declinacion = 23.45 * math.sin(math.radians((360 / 365) * (dia_anio - 81)))

        # Ángulo horario (ω): negativo antes del mediodía, positivo después
        angulo_horario = 15.0 * (hora - 12.0)

        lat_rad = math.radians(latitud)
        dec_rad = math.radians(declinacion)
        h_rad   = math.radians(angulo_horario)

        # Coseno del ángulo cenital solar (cos θz)
        cos_z = (math.sin(lat_rad) * math.sin(dec_rad) +
                 math.cos(lat_rad) * math.cos(dec_rad) * math.cos(h_rad))

        if cos_z > 0.02:   # El sol está sobre el horizonte
            # Factor de corrección por excentricidad orbital terrestre
            ecc     = 1.0 + 0.033 * math.cos(math.radians(360 * dia_anio / 365))
            I_ext   = 1361 * ecc * cos_z          # W/m² extraterrestre

            # Transmisividad alta en Atacama (altitud + baja humedad)
            tau     = 0.80
            ghi_cs  = max(0.0, I_ext * tau)
        else:
            ghi_cs = 0.0

        ghi_list.append(ghi_cs)

    return np.array(ghi_list)


# ============================================================
# 2. CARGA, LIMPIEZA Y PREPARACIÓN DE DATOS
# ============================================================
def preparar_datos():
    print("=" * 60)
    print("  PASO 1: OBTENCIÓN DE DATOS REALES (Meteostat)")
    print("=" * 60)

    df_raw = get_solar_data(dias_historial=365)

    if df_raw is None or df_raw.empty:
        print("\n  [ERROR] No se pudieron obtener datos reales. Verifica tu conexion o la API.")
        sys.exit(1)

    # Resetear índice para tener 'fecha_hora' como columna
    df = df_raw.reset_index().rename(columns={"time": "fecha_hora"})

    # ----------------------------------------------------------
    # Renombramos columnas de Meteostat a nombres en español
    # ----------------------------------------------------------
    renombrar = {
        "temp": "temperatura_c",
        "rhum": "humedad_relativa_pct",
        "wspd": "velocidad_viento_kmh",
        "pres": "presion_hpa",
        "tsun": "sol_minutos_hora",   # minutos de sol por hora (clave para GHI)
        "prcp": "precipitacion_mm",
    }
    df = df.rename(columns=renombrar)

    # Conservamos solo las columnas que usaremos
    columnas_utiles = ["fecha_hora", "temperatura_c", "humedad_relativa_pct",
                       "velocidad_viento_kmh", "sol_minutos_hora"]
    # Incluir presion y precipitación solo si existen
    for col in ["presion_hpa", "precipitacion_mm"]:
        if col in df.columns:
            columnas_utiles.append(col)

    df = df[[c for c in columnas_utiles if c in df.columns]].copy()

    # ----------------------------------------------------------
    # CÁLCULO DE RADIACIÓN SOLAR REAL
    # Combinamos el modelo astronómico con los datos reales de sol
    # tsun: minutos de sol registrados en esa hora (0-60)
    # fraccion_sol: qué fracción de la hora estuvo despejado el cielo
    # GHI_real = GHI_clearsky × fraccion_sol
    # ----------------------------------------------------------
    print("  PASO 2: Calculando Irradiancia Solar (modelo astronómico + datos reales)...")
    ghi_clearsky = calcular_ghi_clearsky(df["fecha_hora"], LATITUD_PARQUE)

    if "sol_minutos_hora" in df.columns and df["sol_minutos_hora"].notna().sum() > 100:
        # Usamos los minutos de sol reales de la estación
        df["sol_minutos_hora"] = df["sol_minutos_hora"].fillna(0).clip(0, 60)
        fraccion_sol = df["sol_minutos_hora"] / 60.0
        print(f"  [OK] Usando 'tsun' real de Meteostat ({df['sol_minutos_hora'].notna().sum()} registros validos)")
    else:
        # Si tsun no tiene datos suficientes, la estimamos desde la hora del día
        # (Atacama tiene ~300 días de sol por año, fracción base alta)
        hora = df["fecha_hora"].dt.hour
        es_dia = (hora >= 7) & (hora <= 18)
        fraccion_sol = np.where(es_dia, 0.85, 0.0)
        print("  [!] 'tsun' con pocos datos, estimando fraccion solar por hora del dia")

    df["radiacion_solar_wm2"] = ghi_clearsky * fraccion_sol
    df["radiacion_solar_wm2"] = df["radiacion_solar_wm2"].clip(lower=0)

    # ----------------------------------------------------------
    # ÍNDICE UV estimado desde GHI
    # En Atacama (2377 m.s.n.m.) el UV es ~25% más alto que a nivel del mar
    # Factor altitud: ~10% extra por cada 1000 m → 2.377 × 10% ≈ 24%
    # ----------------------------------------------------------
    factor_altitud = 1.24
    df["indice_uv"] = (df["radiacion_solar_wm2"] / 80.0) * factor_altitud
    df["indice_uv"] = df["indice_uv"].clip(lower=0)

    # ----------------------------------------------------------
    # COBERTURA DE NUBES estimada (inversa de la fracción de sol)
    # ----------------------------------------------------------
    df["cobertura_nubes_pct"] = (1 - fraccion_sol) * 100

    # ----------------------------------------------------------
    # VARIABLE OBJETIVO: generacion_mw
    # Física real del parque solar:
    #   P = GHI × Área × Eficiencia × Factor_temperatura
    # Factor temperatura: los paneles pierden ~0.4% de eficiencia
    # por cada grado por encima de 25°C (coeficiente típico Si monocristalino)
    # ----------------------------------------------------------
    factor_temp = np.where(
        df["temperatura_c"] > 25,
        1 - (df["temperatura_c"] - 25) * 0.004,
        1.0
    )
    df["generacion_mw"] = (df["radiacion_solar_wm2"] * AREA_PANELES * EFICIENCIA * factor_temp) / 1_000_000
    df["generacion_mw"]  = df["generacion_mw"].clip(lower=0)

    # ----------------------------------------------------------
    # Limpieza final: eliminamos filas con nulos en variables clave
    # ----------------------------------------------------------
    vars_clave = ["temperatura_c", "humedad_relativa_pct", "radiacion_solar_wm2", "generacion_mw"]
    antes = len(df)
    df = df.dropna(subset=vars_clave).reset_index(drop=True)
    print(f"  [OK] Filas despues de limpieza: {len(df)} (eliminadas: {antes - len(df)} con nulos)\n")

    return df


# ============================================================
# 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================
def realizar_eda(df):
    print("=" * 60)
    print("  PASO 3: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 60)

    # Seleccionamos las variables numéricas principales para el análisis
    vars_eda = ["temperatura_c", "humedad_relativa_pct", "cobertura_nubes_pct",
                "velocidad_viento_kmh", "radiacion_solar_wm2", "indice_uv", "generacion_mw"]
    vars_eda = [v for v in vars_eda if v in df.columns]
    df_eda   = df[vars_eda].copy()

    # Estadísticas descriptivas básicas
    print("\nEstadísticas descriptivas:")
    print(df_eda.describe().round(2))

    # ----- 3.1 HEATMAP DE CORRELACIONES -----
    # La correlación mide la relación lineal entre dos variables (−1 a +1)
    # Esperamos que radiacion_solar_wm2 e indice_uv tengan alta correlación con generacion_mw
    plt.figure(figsize=(10, 8))
    matriz_corr = df_eda.corr()
    mascara = np.triu(np.ones_like(matriz_corr, dtype=bool))  # solo triángulo inferior
    sns.heatmap(matriz_corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, mask=mascara, linewidths=0.5)
    plt.title("Mapa de Calor de Correlaciones\n(Parque Solar Calama III - Datos Reales Meteostat)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "01_heatmap_correlaciones.png"), dpi=150)
    plt.show()
    print("  [Gráfico guardado] 01_heatmap_correlaciones.png")

    # ----- 3.2 SCATTER PLOTS -----
    # Relación directa entre las variables predictoras y la generación
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    pares = [
        ("radiacion_solar_wm2", "Radiación Solar (W/m²)", "#E94560"),
        ("indice_uv",           "Índice UV",              "#533483"),
        ("temperatura_c",       "Temperatura (°C)",       "#0F3460"),
        ("cobertura_nubes_pct", "Cobertura de Nubes (%)", "#16213E"),
    ]
    for ax, (var, etiqueta, color) in zip(axs.flat, pares):
        if var in df_eda.columns:
            ax.scatter(df_eda[var], df_eda["generacion_mw"],
                       alpha=0.3, s=8, color=color)
            ax.set_xlabel(etiqueta, fontsize=11)
            ax.set_ylabel("Generación (MW)", fontsize=11)
            ax.set_title(f"Generación vs {etiqueta}", fontsize=12, fontweight="bold")

    plt.suptitle("Scatter Plots: Variables Predictoras vs Generación MW",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "02_scatter_plots.png"), dpi=150)
    plt.show()
    print("  [Gráfico guardado] 02_scatter_plots.png")

    # ----- 3.3 HISTOGRAMAS -----
    # La distribución de cada variable nos indica si hay sesgos o datos atípicos
    vars_hist = ["temperatura_c", "radiacion_solar_wm2", "indice_uv", "generacion_mw"]
    vars_hist = [v for v in vars_hist if v in df_eda.columns]
    fig, axs  = plt.subplots(2, 2, figsize=(14, 9))
    colores_hist = ["#E94560", "#0F3460", "#533483", "#16213E"]
    for ax, var, color in zip(axs.flat, vars_hist, colores_hist):
        sns.histplot(df_eda[var].dropna(), kde=True, bins=40, ax=ax, color=color)
        ax.set_title(f"Distribución: {var}", fontsize=11, fontweight="bold")
        ax.set_xlabel(var)
    plt.suptitle("Histogramas de Variables Principales", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "03_histogramas.png"), dpi=150)
    plt.show()
    print("  [Gráfico guardado] 03_histogramas.png")

    # ----- 3.4 BOXPLOTS -----
    # Detectamos valores atípicos (outliers) por fuera de los bigotes
    plt.figure(figsize=(12, 6))
    df_box = df_eda[["temperatura_c", "velocidad_viento_kmh", "generacion_mw"]].melt(
        var_name="Variable", value_name="Valor"
    )
    sns.boxplot(x="Variable", y="Valor", data=df_box,
                hue="Variable", palette=["#E94560", "#0F3460", "#533483"], legend=False)
    plt.title("Boxplots: Detección de Valores Atípicos (Outliers)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "04_boxplots.png"), dpi=150)
    plt.show()
    print("  [Gráfico guardado] 04_boxplots.png\n")


# ============================================================
# 4. ANÁLISIS DE MULTICOLINEALIDAD (VIF)
# ============================================================
def calcular_vif(X):
    """
    El Factor de Inflación de la Varianza (VIF) mide cuánto aumenta la
    varianza de un coeficiente de regresión por la colinealidad con otras variables.
    Regla general:
        VIF < 5  → Aceptable
        5-10     → Multicolinealidad moderada
        > 10     → Alta multicolinealidad (considerar eliminar la variable)
    """
    df_vif = pd.DataFrame()
    df_vif["Variable"] = X.columns
    df_vif["VIF"]      = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return df_vif.sort_values("VIF", ascending=False).reset_index(drop=True)


# ============================================================
# 5. ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ============================================================
def entrenar_modelos(df):
    print("=" * 60)
    print("  PASO 4: MODELADO PREDICTIVO")
    print("=" * 60)

    # Variables independientes (X) y dependiente (y)
    vars_modelo = ["temperatura_c", "humedad_relativa_pct", "cobertura_nubes_pct",
                   "velocidad_viento_kmh", "radiacion_solar_wm2", "indice_uv"]
    vars_modelo  = [v for v in vars_modelo if v in df.columns]
    X = df[vars_modelo].dropna()
    y = df.loc[X.index, "generacion_mw"]

    # 4.1 ─ Análisis de Multicolinealidad (VIF)
    print("\n  [VIF] Factor de Inflación de Varianza:")
    df_vif = calcular_vif(X)
    print(df_vif.to_string(index=False))

    # 4.2 ─ Feature Selection con SelectKBest (F-test de ANOVA)
    # Seleccionamos las 4 variables con mayor poder predictivo estadístico
    selector = SelectKBest(score_func=f_regression, k=min(4, len(vars_modelo)))
    selector.fit(X, y)
    vars_elegidas = X.columns[selector.get_support()].tolist()
    print(f"\n  [SelectKBest] Variables seleccionadas: {vars_elegidas}")
    X = X[vars_elegidas]

    # 4.3 ─ División Train/Test (80% entrenamiento, 20% prueba)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n  Train: {len(X_train)} registros | Test: {len(X_test)} registros")

    # 4.4 ─ Estandarización (StandardScaler)
    # Esencial para Ridge y Lasso, ya que penalizan los coeficientes por magnitud
    scaler       = StandardScaler()
    X_train_sc   = scaler.fit_transform(X_train)
    X_test_sc    = scaler.transform(X_test)

    # 4.5 ─ Modelos a comparar
    modelos = {
        "Regresión Lineal Múltiple": LinearRegression(),
        "Regresión Ridge (L2)":      Ridge(alpha=1.0),
        "Regresión Lasso (L1)":      Lasso(alpha=0.1, max_iter=5000),
    }

    resultados   = []
    predicciones = {}

    for nombre, modelo in modelos.items():
        # Validación cruzada con 5 pliegues (Cross Validation)
        cv_r2 = cross_val_score(modelo, X_train_sc, y_train, cv=5, scoring="r2")

        # Entrenamiento final sobre todo el set de entrenamiento
        modelo.fit(X_train_sc, y_train)
        y_pred = modelo.predict(X_test_sc)
        y_pred = np.clip(y_pred, 0, None)   # Generación no puede ser negativa

        predicciones[nombre] = y_pred

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)

        resultados.append({
            "Modelo":      nombre,
            "CV R² (media)": round(cv_r2.mean(), 4),
            "MAE":         round(mae, 3),
            "RMSE":        round(rmse, 3),
            "R² Test":     round(r2, 4),
        })

    # Tabla comparativa de resultados
    df_resultados = pd.DataFrame(resultados)
    print("\n  --- COMPARACIÓN DE MODELOS ---")
    print(df_resultados.to_string(index=False))

    # 4.6 ─ Análisis de Residuos (modelo lineal como referencia)
    print("\n  [Gráfico] Generando análisis de residuos...")
    y_pred_lineal = predicciones["Regresión Lineal Múltiple"]
    residuos      = np.array(y_test) - y_pred_lineal

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    axs[0].scatter(y_pred_lineal, residuos, alpha=0.4, s=10, color="#E94560")
    axs[0].axhline(0, color="black", linewidth=1.5, linestyle="--")
    axs[0].set_xlabel("Predicciones (MW)", fontsize=11)
    axs[0].set_ylabel("Residuos (Real - Predicción)", fontsize=11)
    axs[0].set_title("Residuos vs Predicciones\n(Regresión Lineal)", fontsize=12, fontweight="bold")

    # QQ-plot manual: ¿los residuos siguen una distribución normal?
    from scipy import stats
    stats.probplot(residuos, dist="norm", plot=axs[1])
    axs[1].set_title("Q-Q Plot de Residuos\n(Normalidad de errores)", fontsize=12, fontweight="bold")
    axs[1].get_lines()[0].set(color="#E94560", markersize=3, alpha=0.5)
    axs[1].get_lines()[1].set(color="#0F3460", linewidth=2)

    plt.suptitle("Análisis de Residuos - Regresión Lineal Múltiple",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "05_analisis_residuos.png"), dpi=150)
    plt.show()
    print("  [Gráfico guardado] 05_analisis_residuos.png")

    # 4.7 ─ Gráfico Real vs Predicho
    plt.figure(figsize=(9, 7))
    plt.scatter(y_test, y_pred_lineal, alpha=0.4, s=10, color="#533483", label="Predicciones")
    lim_max = max(y_test.max(), y_pred_lineal.max()) * 1.05
    plt.plot([0, lim_max], [0, lim_max], "r--", linewidth=2, label="Línea perfecta")
    plt.xlabel("Generación Real (MW)", fontsize=12)
    plt.ylabel("Generación Predicha (MW)", fontsize=12)
    plt.title("Real vs Predicho -- Regresión Lineal Múltiple", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GRAFICOS, "06_real_vs_predicho.png"), dpi=150)
    plt.show()
    print("  [Gráfico guardado] 06_real_vs_predicho.png\n")

    return df_resultados


# ============================================================
# PUNTO DE ENTRADA PRINCIPAL
# ============================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  PROYECTO DS: PROYECCIÓN SOLAR -- PARQUE CALAMA III")
    print("  Datos reales: Meteostat | Atacama, Chile")
    print("=" * 60 + "\n")

    # 1. Obtener y preparar datos reales
    df_final = preparar_datos()

    # 2. Análisis Exploratorio
    realizar_eda(df_final)

    # 3. Modelado y evaluación
    df_metricas = entrenar_modelos(df_final)

    print("=" * 60)
    print("  [OK] PROYECTO COMPLETADO")
    print(f"  Graficos guardados en: visualizaciones/")
    print("=" * 60 + "\n")
