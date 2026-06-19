# ============================================================================
#  preparar_datos.py
#  ---------------------------------------------------------------------------
#  Convierte los archivos que produce tu notebook de Colab en archivos
#  pequeños y listos para que la página web (app.py) los muestre rápido.
#
#  ENTRADA  (carpeta fuente_modelo/, generada por tu notebook):
#     - suicidios_01_preprocesado.csv    (hechos a nivel registro)
#     - suicidios_03_panel_modelado.csv  (panel provincia x mes con features)
#     - modelo_alerta_temprana.joblib    (modelo Random Forest + SMOTE)
#
#  SALIDA  (carpeta data/, que lee la app):
#     - tablero_provincias.csv      probabilidad de alerta del proximo mes
#     - serie_mensual.csv           casos por provincia y mes
#     - resumen_provincia.csv       totales y promedios por provincia
#     - resumen_demografico.csv     distribuciones (sexo, edad, modalidad)
#     - metricas_modelos.csv        comparativa de los modelos evaluados
#     - importancia_variables.csv   importancia de cada variable del modelo
#
#  USO:   python preparar_datos.py
# ============================================================================
import os
import numpy as np
import pandas as pd
import joblib

FUENTE = "fuente_modelo"
SALIDA = "data"
os.makedirs(SALIDA, exist_ok=True)

# ---------------------------------------------------------------------------
# Coordenadas (capital / centro aproximado) de cada provincia, para el mapa
# ---------------------------------------------------------------------------
COORDS = {
    "Buenos Aires": (-36.50, -59.80),
    "Ciudad Autónoma de Buenos Aires": (-34.61, -58.38),
    "Catamarca": (-28.47, -65.78),
    "Chaco": (-27.45, -58.99),
    "Chubut": (-43.30, -65.10),
    "Córdoba": (-31.42, -64.18),
    "Corrientes": (-27.47, -58.83),
    "Entre Ríos": (-31.73, -60.53),
    "Formosa": (-26.18, -58.17),
    "Jujuy": (-24.19, -65.30),
    "La Pampa": (-36.62, -64.29),
    "La Rioja": (-29.41, -66.86),
    "Mendoza": (-32.89, -68.84),
    "Misiones": (-27.36, -55.90),
    "Neuquén": (-38.95, -68.06),
    "Río Negro": (-40.81, -63.00),
    "Salta": (-24.79, -65.41),
    "San Juan": (-31.54, -68.54),
    "San Luis": (-33.30, -66.34),
    "Santa Cruz": (-51.62, -69.22),
    "Santa Fe": (-31.63, -60.70),
    "Santiago del Estero": (-27.80, -64.26),
    "Tucumán": (-26.82, -65.22),
}

# Umbrales del semáforo (probabilidad de alerta)
UMBRAL_ALTO = 0.60
UMBRAL_MEDIO = 0.35


def estado_desde_prob(p):
    if p >= UMBRAL_ALTO:
        return "Alerta alta", 3
    if p >= UMBRAL_MEDIO:
        return "Atención", 2
    return "Normal", 1


print("1/5  Cargando insumos...")
preproc = pd.read_csv(os.path.join(FUENTE, "suicidios_01_preprocesado.csv"))
panel = pd.read_csv(os.path.join(FUENTE, "suicidios_03_panel_modelado.csv"))
paquete = joblib.load(os.path.join(FUENTE, "modelo_alerta_temprana.joblib"))
pipeline = paquete["pipeline"]
features = paquete["features"]

# ---------------------------------------------------------------------------
# 2) TABLERO: probabilidad de alerta del proximo mes por provincia
# ---------------------------------------------------------------------------
print("2/5  Calculando probabilidad de alerta por provincia...")


def features_proximo_mes(panel, provincia):
    """Construye la fila del 'proximo mes' a partir del historial reciente."""
    sub = panel[panel["provincia_nombre"] == provincia].sort_values(["anio", "mes"])
    ult = sub.iloc[-1]
    casos = sub["casos"].values
    fila = {
        "lag1": casos[-1],
        "lag2": casos[-2],
        "lag3": casos[-3],
        "roll3_mean": casos[-3:].mean(),
        "roll3_std": casos[-3:].std(),
        "prop_masc": ult["prop_masc"],
        "prop_joven": ult["prop_joven"],
        "edad_media": ult["edad_media"],
        "mes": (int(ult["mes"]) % 12) + 1,
        "region": ult["region"],
        "cluster_territorial": ult["cluster_territorial"],
        "perfil_dominante": ult["perfil_dominante"],
    }
    return pd.DataFrame([fila])[features]


filas = []
for prov in sorted(panel["provincia_nombre"].unique()):
    X = features_proximo_mes(panel, prov)
    prob = float(pipeline.predict_proba(X)[0, 1])
    estado, orden = estado_desde_prob(prob)
    region = panel[panel["provincia_nombre"] == prov]["region"].iloc[-1]
    lat, lon = COORDS.get(prov, (np.nan, np.nan))
    casos_total = int(panel[panel["provincia_nombre"] == prov]["casos"].sum())
    filas.append({
        "provincia": prov, "region": region, "lat": lat, "lon": lon,
        "prob_alerta": round(prob, 4), "prob_pct": round(prob * 100, 1),
        "estado": estado, "nivel_orden": orden, "casos_historico": casos_total,
    })

tablero = pd.DataFrame(filas).sort_values("prob_alerta", ascending=False)
tablero.to_csv(os.path.join(SALIDA, "tablero_provincias.csv"), index=False)
print(f"      -> tablero_provincias.csv ({len(tablero)} provincias)")

# ---------------------------------------------------------------------------
# 3) SERIE MENSUAL de casos por provincia (para graficos de evolucion)
# ---------------------------------------------------------------------------
print("3/5  Construyendo serie mensual...")
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
print("4/5  Resumiendo descriptivos...")
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

# distribuciones demograficas (formato largo: dimension, categoria, casos)
bloques = []
for col, dim in [("suicida_sexo", "Sexo"),
                 ("grupo_etario", "Grupo etario"),
                 ("modalidad", "Modalidad"),
                 ("estacion", "Estación"),
                 ("franja_horaria", "Franja horaria")]:
    if col in preproc.columns:
        vc = preproc[col].value_counts().reset_index()
        vc.columns = ["categoria", "casos"]
        vc.insert(0, "dimension", dim)
        bloques.append(vc)
demo = pd.concat(bloques, ignore_index=True)
demo.to_csv(os.path.join(SALIDA, "resumen_demografico.csv"), index=False)
print(f"      -> resumen_provincia.csv / resumen_demografico.csv")

# ---------------------------------------------------------------------------
# 5) METRICAS de modelos e IMPORTANCIA de variables
# ---------------------------------------------------------------------------
print("5/5  Exportando metricas e importancia...")
# Metricas observadas en la evaluacion del notebook (conjunto de prueba 2021-2022)
metricas = pd.DataFrame([
    ["Regresión Logística", False, 0.755, 0.500, 0.007, 0.015, 0.578, 0.316],
    ["Regresión Logística", True,  0.551, 0.260, 0.452, 0.330, 0.526, 0.284],
    ["Random Forest",       False, 0.757, 1.000, 0.007, 0.015, 0.704, 0.443],
    ["Random Forest",       True,  0.679, 0.379, 0.489, 0.427, 0.680, 0.371],
    ["Gradient Boosting",   False, 0.754, 0.471, 0.059, 0.105, 0.703, 0.395],
    ["Gradient Boosting",   True,  0.716, 0.420, 0.430, 0.425, 0.705, 0.409],
    ["XGBoost",             False, 0.732, 0.362, 0.126, 0.187, 0.686, 0.363],
    ["XGBoost",             True,  0.741, 0.460, 0.341, 0.391, 0.715, 0.419],
], columns=["modelo", "usa_smote", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"])
metricas.to_csv(os.path.join(SALIDA, "metricas_modelos.csv"), index=False)

# Importancia de variables del Random Forest final
try:
    clf = pipeline.named_steps["clf"]
    prep = pipeline.named_steps["prep"]
    nombres = prep.get_feature_names_out()
    nombres = [n.split("__")[-1] for n in nombres]
    imp = (pd.DataFrame({"variable": nombres, "importancia": clf.feature_importances_})
           .sort_values("importancia", ascending=False).head(15))
    imp["importancia"] = imp["importancia"].round(4)
    imp.to_csv(os.path.join(SALIDA, "importancia_variables.csv"), index=False)
    print(f"      -> metricas_modelos.csv / importancia_variables.csv")
except Exception as e:
    print("      (no se pudo extraer importancia:", e, ")")

print("\nLISTO. Archivos generados en la carpeta data/")
