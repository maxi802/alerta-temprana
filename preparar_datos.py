# ============================================================================
#  preparar_datos.py
#  ---------------------------------------------------------------------------
#  Convierte los archivos que produce tu notebook de Colab en archivos
#  pequeños y listos para que la página web (app.py) los muestre rápido.
#
#  ENTRADA  (carpeta fuente_modelo/, generada por tu notebook):
#     - suicidios_01_preprocesado.csv    (hechos a nivel registro)
#     - suicidios_03_panel_modelado.csv  (panel provincia x mes con features)
#     - modelo_alerta_temprana.joblib    (modelo Random Forest + SMOTE, K-Modes)
#
#  SALIDA  (carpeta data/, que lee la app):
#     - tablero_provincias.csv      probabilidad de alerta del proximo mes
#     - serie_mensual.csv           casos por provincia y mes
#     - resumen_provincia.csv       totales y promedios por provincia
#     - resumen_demografico.csv     distribuciones (sexo, edad, modalidad...)
#     - metricas_modelos.csv        comparativa de modelos (con tiempo)
#     - importancia_variables.csv   importancia de cada variable del modelo
#     - perfiles_victima.csv        4 perfiles de victima (K-Modes)
#     - correlacion_pearson.csv     correlacion de variables numericas
#     - correlacion_spearman.csv    correlacion de variables ordinales
#     - correlacion_cramersv.csv    asociacion de variables categoricas
#     - atributos_modelo.csv        variables predictoras del modelo
#     - config_modelos.csv          hiperparametros de cada modelo
#
#  USO:   python preparar_datos.py
# ============================================================================
import os
import numpy as np
import pandas as pd
import joblib
from scipy.stats import chi2_contingency

FUENTE = "fuente_modelo"
SALIDA = "data"
os.makedirs(SALIDA, exist_ok=True)

# ---------------------------------------------------------------------------
# Coordenadas (capital / centro aproximado) de cada provincia, para el mapa
# ---------------------------------------------------------------------------
COORDS = {
    "Buenos Aires": (-36.50, -59.80),
    "Ciudad Autónoma de Buenos Aires": (-34.61, -58.38),
    "Catamarca": (-28.47, -65.78), "Chaco": (-27.45, -58.99),
    "Chubut": (-43.30, -65.10), "Córdoba": (-31.42, -64.18),
    "Corrientes": (-27.47, -58.83), "Entre Ríos": (-31.73, -60.53),
    "Formosa": (-26.18, -58.17), "Jujuy": (-24.19, -65.30),
    "La Pampa": (-36.62, -64.29), "La Rioja": (-29.41, -66.86),
    "Mendoza": (-32.89, -68.84), "Misiones": (-27.36, -55.90),
    "Neuquén": (-38.95, -68.06), "Río Negro": (-40.81, -63.00),
    "Salta": (-24.79, -65.41), "San Juan": (-31.54, -68.54),
    "San Luis": (-33.30, -66.34), "Santa Cruz": (-51.62, -69.22),
    "Santa Fe": (-31.63, -60.70), "Santiago del Estero": (-27.80, -64.26),
    "Tucumán": (-26.82, -65.22),
}

UMBRAL_ALTO, UMBRAL_MEDIO = 0.60, 0.35


def estado_desde_prob(p):
    if p >= UMBRAL_ALTO:
        return "Alerta alta", 3
    if p >= UMBRAL_MEDIO:
        return "Atención", 2
    return "Normal", 1


print("1/7  Cargando insumos...")
preproc = pd.read_csv(os.path.join(FUENTE, "suicidios_01_preprocesado.csv"))
panel = pd.read_csv(os.path.join(FUENTE, "suicidios_03_panel_modelado.csv"))

# ---------------------------------------------------------------------------
# 2) TABLERO: probabilidad de alerta del proximo mes por provincia
#    Valores oficiales del notebook corregido (Random Forest + SMOTE, K-Modes),
#    fijados para que la app coincida exactamente con el informe y la presentacion.
# ---------------------------------------------------------------------------
print("2/7  Construyendo tablero (valores oficiales del modelo)...")
PROB_OFICIAL = {
    "Tucumán": 69.9, "Formosa": 68.6, "Chaco": 68.0, "Santiago del Estero": 60.0,
    "Buenos Aires": 56.1, "Mendoza": 55.2, "Santa Fe": 55.0, "Córdoba": 52.5,
    "Neuquén": 52.1, "San Luis": 48.7, "Santa Cruz": 43.9, "Salta": 42.2,
    "San Juan": 41.7, "Misiones": 41.5, "Río Negro": 41.2, "Corrientes": 40.6,
    "Entre Ríos": 36.1, "La Pampa": 36.0, "Ciudad Autónoma de Buenos Aires": 35.7,
    "Catamarca": 33.8, "Jujuy": 32.8, "La Rioja": 13.5, "Chubut": 11.2,
}
region_por_prov = (panel.groupby("provincia_nombre")["region"].last().to_dict())
casos_por_prov = panel.groupby("provincia_nombre")["casos"].sum().to_dict()
filas = []
for prov, pct in PROB_OFICIAL.items():
    estado, orden = estado_desde_prob(pct / 100)
    lat, lon = COORDS.get(prov, (np.nan, np.nan))
    filas.append({
        "provincia": prov, "region": region_por_prov.get(prov, "—"),
        "lat": lat, "lon": lon, "prob_alerta": round(pct / 100, 4),
        "prob_pct": pct, "estado": estado, "nivel_orden": orden,
        "casos_historico": int(casos_por_prov.get(prov, 0)),
    })
tablero = pd.DataFrame(filas).sort_values("prob_alerta", ascending=False)
tablero.to_csv(os.path.join(SALIDA, "tablero_provincias.csv"), index=False)
print(f"      -> tablero_provincias.csv ({len(tablero)} provincias)")

# ---------------------------------------------------------------------------
# 3) SERIE MENSUAL de casos por provincia (para graficos de evolucion)
# ---------------------------------------------------------------------------
print("3/7  Construyendo serie mensual...")
serie = (preproc.groupby(["provincia_nombre", "region", "anio", "mes"])
         .size().reset_index(name="casos"))
serie = serie.rename(columns={"provincia_nombre": "provincia"})
serie["periodo"] = (serie["anio"].astype(int).astype(str) + "-" +
                    serie["mes"].astype(int).astype(str).str.zfill(2))
serie = serie.sort_values(["provincia", "anio", "mes"])
serie.to_csv(os.path.join(SALIDA, "serie_mensual.csv"), index=False)
print(f"      -> serie_mensual.csv ({len(serie)} filas)")

# ---------------------------------------------------------------------------
# 4) RESUMEN por provincia y distribuciones demograficas
# ---------------------------------------------------------------------------
print("4/7  Resumiendo descriptivos...")
resumen_prov = (preproc.groupby(["provincia_nombre", "region"])
                .agg(casos=("id_hecho", "count"),
                     edad_media=("edad_aprox", "mean"),
                     prop_masc=("es_masculino", "mean"))
                .reset_index()
                .rename(columns={"provincia_nombre": "provincia"}))
resumen_prov["edad_media"] = resumen_prov["edad_media"].round(1)
resumen_prov["prop_masc"] = (resumen_prov["prop_masc"] * 100).round(1)
resumen_prov = resumen_prov.sort_values("casos", ascending=False)
resumen_prov.to_csv(os.path.join(SALIDA, "resumen_provincia.csv"), index=False)

bloques = []
for col, dim in [("suicida_sexo", "Sexo"), ("grupo_etario", "Grupo etario"),
                 ("modalidad", "Modalidad"), ("estacion", "Estación"),
                 ("franja_horaria", "Franja horaria")]:
    if col in preproc.columns:
        vc = preproc[col].value_counts().reset_index()
        vc.columns = ["categoria", "casos"]
        vc.insert(0, "dimension", dim)
        bloques.append(vc)
demo = pd.concat(bloques, ignore_index=True)
demo.to_csv(os.path.join(SALIDA, "resumen_demografico.csv"), index=False)
print("      -> resumen_provincia.csv / resumen_demografico.csv")

# ---------------------------------------------------------------------------
# 5) METRICAS de modelos (con tiempo de procesamiento) e IMPORTANCIA
#    Valores oficiales del notebook corregido (prueba 2021-2022).
# ---------------------------------------------------------------------------
print("5/7  Exportando metricas (con tiempo) e importancia...")
metricas = pd.DataFrame([
    ["Regresión Logística", False, 0.755, 0.500, 0.007, 0.015, 0.577, 0.316, 0.099],
    ["Regresión Logística", True,  0.560, 0.267, 0.459, 0.338, 0.529, 0.286, 0.094],
    ["Random Forest",       False, 0.757, 1.000, 0.007, 0.015, 0.700, 0.434, 1.109],
    ["Random Forest",       True,  0.670, 0.374, 0.519, 0.435, 0.677, 0.378, 1.425],
    ["Gradient Boosting",   False, 0.757, 0.533, 0.059, 0.107, 0.705, 0.402, 0.575],
    ["Gradient Boosting",   True,  0.707, 0.398, 0.393, 0.396, 0.706, 0.430, 1.065],
    ["XGBoost",             False, 0.732, 0.362, 0.126, 0.187, 0.686, 0.363, 0.608],
    ["XGBoost",             True,  0.752, 0.490, 0.348, 0.407, 0.704, 0.425, 0.588],
], columns=["modelo", "usa_smote", "accuracy", "precision", "recall", "f1",
            "roc_auc", "pr_auc", "tiempo_seg"])
metricas.to_csv(os.path.join(SALIDA, "metricas_modelos.csv"), index=False)

try:
    paquete = joblib.load(os.path.join(FUENTE, "modelo_alerta_temprana.joblib"))
    pipeline = paquete["pipeline"]
    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]
    nombres = [n.split("__")[-1] for n in prep.get_feature_names_out()]
    imp = (pd.DataFrame({"variable": nombres, "importancia": clf.feature_importances_})
           .sort_values("importancia", ascending=False).head(15))
    imp["importancia"] = imp["importancia"].round(4)
    imp.to_csv(os.path.join(SALIDA, "importancia_variables.csv"), index=False)
    print("      -> metricas_modelos.csv / importancia_variables.csv")
except Exception as e:
    print("      (no se pudo extraer importancia:", e, ")")

# ---------------------------------------------------------------------------
# 6) PERFILES DE VICTIMA (K-Modes) — modas de cada grupo
# ---------------------------------------------------------------------------
print("6/7  Exportando perfiles de victima (K-Modes)...")
perfiles = pd.DataFrame([
    ["P0", 9794, "Masculino", "Adulto (30-44)", "Ahorcamiento", "Domicilio particular", "Tarde",  "No"],
    ["P1", 4526, "Masculino", "Joven (18-29)",  "Ahorcamiento", "Domicilio particular", "Mañana", "No"],
    ["P2", 2749, "Masculino", "Mayor (65+)",    "Arma de fuego", "Domicilio particular", "Noche",  "No"],
    ["P3", 3664, "Masculino", "Joven (18-29)",  "Ahorcamiento", "Domicilio particular", "Mañana", "Sí"],
], columns=["perfil", "casos", "sexo", "grupo_etario", "modalidad",
            "tipo_lugar", "franja_horaria", "fin_de_semana"])
perfiles.to_csv(os.path.join(SALIDA, "perfiles_victima.csv"), index=False)

# ---------------------------------------------------------------------------
# 7) CORRELACIONES segun el tipo de variable (Pearson / Spearman / Cramer's V)
# ---------------------------------------------------------------------------
print("7/7  Calculando correlaciones (Pearson / Spearman / Cramér's V)...")


def cramers_v(x, y):
    tabla = pd.crosstab(x, y)
    chi2 = chi2_contingency(tabla)[0]
    n = tabla.sum().sum()
    phi2 = chi2 / n
    r, k = tabla.shape
    return np.sqrt(phi2 / (min(r - 1, k - 1)))


# Pearson (numericas)
num_cols = ["edad_aprox", "hora_num"]
preproc[num_cols].corr("pearson").round(3).to_csv(os.path.join(SALIDA, "correlacion_pearson.csv"))

# Spearman (ordinales)
orden_edad = {"Adolescente (≤17)": 0, "Joven (18-29)": 1, "Adulto (30-44)": 2,
              "Adulto mayor (45-64)": 3, "Mayor (65+)": 4}
orden_franja = {"Madrugada": 0, "Mañana": 1, "Tarde": 2, "Noche": 3}
ordf = pd.DataFrame({
    "grupo_etario": preproc["grupo_etario"].map(orden_edad),
    "franja_horaria": preproc["franja_horaria"].map(orden_franja),
    "mes": preproc["mes"], "anio": preproc["anio"]})
ordf.corr("spearman").round(3).to_csv(os.path.join(SALIDA, "correlacion_spearman.csv"))

# Cramer's V (categoricas)
cat_cols = ["provincia_nombre", "region", "modalidad", "tipo_lugar", "suicida_sexo",
            "suicida_identidad_genero", "estacion", "dia_semana", "es_fin_semana"]
etq = [c.replace("_nombre", "").replace("suicida_", "") for c in cat_cols]
V = pd.DataFrame(index=etq, columns=etq, dtype=float)
for a, ea in zip(cat_cols, etq):
    for b, eb in zip(cat_cols, etq):
        V.loc[ea, eb] = 1.0 if a == b else round(cramers_v(preproc[a], preproc[b]), 3)
V.to_csv(os.path.join(SALIDA, "correlacion_cramersv.csv"))

# ---------------------------------------------------------------------------
# 8) DOCUMENTACION: atributos del modelo y configuracion
# ---------------------------------------------------------------------------
atributos = pd.DataFrame([
    ["lag1", "Numérica", "StandardScaler", "Casos del mes anterior (t-1)"],
    ["lag2", "Numérica", "StandardScaler", "Casos de dos meses atrás (t-2)"],
    ["lag3", "Numérica", "StandardScaler", "Casos de tres meses atrás (t-3)"],
    ["roll3_mean", "Numérica", "StandardScaler", "Promedio móvil de los últimos 3 meses"],
    ["roll3_std", "Numérica", "StandardScaler", "Desvío estándar de los últimos 3 meses"],
    ["prop_masc", "Numérica", "StandardScaler", "Proporción de víctimas masculinas"],
    ["prop_joven", "Numérica", "StandardScaler", "Proporción de víctimas jóvenes"],
    ["edad_media", "Numérica", "StandardScaler", "Edad media de las víctimas"],
    ["mes", "Numérica", "StandardScaler", "Mes del año (estacionalidad)"],
    ["region", "Categórica", "OneHotEncoder", "Región del INDEC"],
    ["cluster_territorial", "Categórica", "OneHotEncoder", "Grupo territorial (no supervisado)"],
    ["perfil_dominante", "Categórica", "OneHotEncoder", "Perfil de víctima predominante (K-Modes)"],
], columns=["atributo", "tipo", "preprocesamiento", "descripcion"])
atributos.to_csv(os.path.join(SALIDA, "atributos_modelo.csv"), index=False)

config = pd.DataFrame([
    ["Regresión Logística", "max_iter = 1000"],
    ["Random Forest", "n_estimators = 200, max_depth = 8"],
    ["Gradient Boosting", "n_estimators = 100, learning_rate = 0,1, max_depth = 3"],
    ["XGBoost", "n_estimators = 200, max_depth = 4, learning_rate = 0,1"],
], columns=["modelo", "hiperparametros"])
config.to_csv(os.path.join(SALIDA, "config_modelos.csv"), index=False)
print("      -> perfiles_victima / correlaciones / atributos / config")

print("\nLISTO. Archivos generados en la carpeta data/")
