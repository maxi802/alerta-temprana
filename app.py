# ============================================================================
#  app.py  —  Sistema de Alerta Temprana de Suicidios (Argentina)
#  ---------------------------------------------------------------------------
#  Tablero web construido con Streamlit + Plotly sobre los resultados del
#  modelo de Machine Learning (Random Forest + SMOTE).
#
#  Para ejecutar:   streamlit run app.py
# ============================================================================
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# Configuración general de la página
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Alerta Temprana · Argentina",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de colores del semáforo
COLOR_ESTADO = {"Alerta alta": "#E63946", "Atención": "#E9A23B", "Normal": "#2A9D8F"}
AZUL = "#1F4E79"

# Estilos propios (tarjetas de indicadores y ajustes visuales)
st.markdown(f"""
<style>
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    h1, h2, h3 {{ color: {AZUL}; }}
    .titulo-app {{ font-size: 2.0rem; font-weight: 800; color: {AZUL}; margin-bottom: .2rem; }}
    .subtitulo-app {{ color: #5A5A5A; font-size: 1.02rem; margin-bottom: 1.2rem; }}
    .kpi {{ border-radius: 14px; padding: 18px 20px; color: white;
            box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
    .kpi .n {{ font-size: 2.3rem; font-weight: 800; line-height: 1; }}
    .kpi .l {{ font-size: .92rem; opacity: .95; margin-top: 6px; }}
    .nota {{ background:#F2F7FC; border:1px solid #BcD3EA; border-left:5px solid {AZUL};
             border-radius:10px; padding:14px 18px; font-size:.9rem; color:#33475b; }}
    [data-testid="stMetricValue"] {{ color: {AZUL}; }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Carga de datos (con caché para que la app sea rápida)
# ----------------------------------------------------------------------------
@st.cache_data
def cargar(nombre):
    return pd.read_csv(f"data/{nombre}")


try:
    tablero = cargar("tablero_provincias.csv")
    serie = cargar("serie_mensual.csv")
    resumen_prov = cargar("resumen_provincia.csv")
    demo = cargar("resumen_demografico.csv")
    metricas = cargar("metricas_modelos.csv")
    importancia = cargar("importancia_variables.csv")
except FileNotFoundError:
    st.error("No se encontraron los archivos de la carpeta **data/**. "
             "Ejecutá primero  `python preparar_datos.py`  para generarlos.")
    st.stop()


# ----------------------------------------------------------------------------
# Encabezado
# ----------------------------------------------------------------------------
st.markdown('<div class="titulo-app">🚨 Sistema de Alerta Temprana de Suicidios</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitulo-app">Monitoreo provincial basado en Ciencia de Datos · '
            'Registro SNIC 2017–2022 · Modelo Random Forest + SMOTE</div>',
            unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Barra lateral: FILTROS
# ----------------------------------------------------------------------------
st.sidebar.header("Filtros")

regiones = sorted(tablero["region"].unique())
sel_region = st.sidebar.multiselect("Región", regiones, default=regiones)

provincias_disp = sorted(tablero[tablero["region"].isin(sel_region)]["provincia"].unique())
sel_provincia = st.sidebar.multiselect(
    "Provincia", provincias_disp, default=provincias_disp,
    help="Por defecto se muestran todas las provincias de las regiones elegidas.")

niveles = ["Alerta alta", "Atención", "Normal"]
sel_nivel = st.sidebar.multiselect("Nivel de alerta", niveles, default=niveles)

anios = sorted(serie["anio"].unique())
rango = st.sidebar.select_slider("Período (años)", options=anios,
                                 value=(anios[0], anios[-1]))

st.sidebar.markdown("---")
st.sidebar.caption("El semáforo se calcula con la probabilidad estimada de que el "
                   "próximo mes supere el nivel habitual de cada provincia:")
st.sidebar.markdown(
    f"- 🔴 **Alerta alta** ≥ 60%\n- 🟠 **Atención** 35–60%\n- 🟢 **Normal** < 35%")

# Aplicar filtros
tb = tablero[
    tablero["region"].isin(sel_region)
    & tablero["provincia"].isin(sel_provincia)
    & tablero["estado"].isin(sel_nivel)
].copy()

sr = serie[
    serie["region"].isin(sel_region)
    & serie["provincia"].isin(sel_provincia)
    & serie["anio"].between(rango[0], rango[1])
].copy()

if tb.empty:
    st.warning("Ningún dato coincide con los filtros seleccionados. Ajustá los filtros en la barra lateral.")
    st.stop()


# ----------------------------------------------------------------------------
# PESTAÑAS
# ----------------------------------------------------------------------------
tab_mapa, tab_evol, tab_eda, tab_modelo = st.tabs(
    ["🗺️  Tablero y mapa", "📈  Evolución temporal", "📊  Análisis de datos", "🤖  El modelo"])


# ============================== TAB 1: MAPA =================================
with tab_mapa:
    # Indicadores
    n_alta = int((tb["estado"] == "Alerta alta").sum())
    n_aten = int((tb["estado"] == "Atención").sum())
    n_norm = int((tb["estado"] == "Normal").sum())
    prob_prom = tb["prob_pct"].mean()

    c1, c2, c3, c4 = st.columns(4)
    for col, n, lbl, color in [
        (c1, n_alta, "Provincias en alerta alta", COLOR_ESTADO["Alerta alta"]),
        (c2, n_aten, "Provincias en atención", COLOR_ESTADO["Atención"]),
        (c3, n_norm, "Provincias en nivel normal", COLOR_ESTADO["Normal"]),
        (c4, f"{prob_prom:.0f}%", "Probabilidad promedio", AZUL),
    ]:
        col.markdown(
            f'<div class="kpi" style="background:{color}"><div class="n">{n}</div>'
            f'<div class="l">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("")
    izq, der = st.columns([1.35, 1])

    # Mapa
    with izq:
        st.subheader("Mapa de riesgo por provincia")
        fig = px.scatter_geo(
            tb, lat="lat", lon="lon", color="estado", size="prob_pct",
            hover_name="provincia", size_max=30,
            color_discrete_map=COLOR_ESTADO,
            category_orders={"estado": niveles},
            hover_data={"lat": False, "lon": False, "region": True,
                        "prob_pct": ":.1f", "estado": True},
            labels={"prob_pct": "Prob. alerta (%)", "region": "Región", "estado": "Estado"})
        fig.update_geos(showcountries=True, showsubunits=True,
                        countrycolor="#A9B6C4", subunitcolor="#CDD6E0",
                        landcolor="#F4F7FB", showland=True, showocean=True,
                        oceancolor="#EAF1F8", resolution=50,
                        projection_type="mercator",
                        lataxis_range=[-56, -21], lonaxis_range=[-77, -52])
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=520,
                          legend=dict(orientation="h", yanchor="bottom", y=0.01,
                                      xanchor="left", x=0.01, title=""))
        st.plotly_chart(fig, width="stretch")

    # Tabla coloreada
    with der:
        st.subheader("Ranking de provincias")

        def color_estado(val):
            return f"background-color:{COLOR_ESTADO.get(val,'#fff')}; color:white;"

        tabla_view = (tb[["provincia", "region", "prob_pct", "estado"]]
                      .rename(columns={"provincia": "Provincia", "region": "Región",
                                       "prob_pct": "Prob. (%)", "estado": "Estado"}))
        st.dataframe(
            tabla_view.style.map(color_estado, subset=["Estado"])
                            .format({"Prob. (%)": "{:.1f}"}),
            width="stretch", height=520, hide_index=True)


# ============================== TAB 2: EVOLUCIÓN ============================
with tab_evol:
    st.subheader("Evolución mensual de casos")
    st.caption("Cantidad de hechos registrados por mes, según las provincias y el período seleccionados.")

    serie_mes = (sr.groupby("periodo", as_index=False)["casos"].sum()
                 .sort_values("periodo"))
    fig2 = px.area(serie_mes, x="periodo", y="casos", markers=False,
                   labels={"periodo": "Mes", "casos": "Casos"})
    fig2.update_traces(line_color=AZUL, fillcolor="rgba(31,78,121,.15)")
    fig2.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, width="stretch")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Casos por año")
        por_anio = sr.groupby("anio", as_index=False)["casos"].sum()
        figA = px.bar(por_anio, x="anio", y="casos", text="casos",
                      labels={"anio": "Año", "casos": "Casos"})
        figA.update_traces(marker_color=AZUL)
        figA.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figA, width="stretch")
    with colB:
        st.subheader("Estacionalidad (por mes)")
        por_mes = sr.groupby("mes", as_index=False)["casos"].sum()
        figB = px.bar(por_mes, x="mes", y="casos",
                      labels={"mes": "Mes", "casos": "Casos"})
        figB.update_traces(marker_color="#2A9D8F")
        figB.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis=dict(tickmode="linear"))
        st.plotly_chart(figB, width="stretch")

    st.subheader("Comparación entre provincias")
    top_prov = (sr.groupby("provincia", as_index=False)["casos"].sum()
                .sort_values("casos", ascending=False).head(15))
    figC = px.bar(top_prov.sort_values("casos"), x="casos", y="provincia",
                  orientation="h", labels={"casos": "Casos", "provincia": ""})
    figC.update_traces(marker_color=AZUL)
    figC.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figC, width="stretch")


# ============================== TAB 3: ANÁLISIS ============================
with tab_eda:
    st.subheader("Distribuciones del conjunto de datos")
    st.caption("Estas distribuciones corresponden al total de hechos analizados (no se ven afectadas por los filtros).")

    dims = demo["dimension"].unique().tolist()
    dim = st.selectbox("Dimensión a explorar", dims, index=0)
    d = demo[demo["dimension"] == dim].sort_values("casos", ascending=True)
    figD = px.bar(d, x="casos", y="categoria", orientation="h",
                  labels={"casos": "Casos", "categoria": ""}, text="casos")
    figD.update_traces(marker_color=AZUL)
    figD.update_layout(height=max(300, 40 * len(d)), margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figD, width="stretch")

    st.subheader("Perfil por provincia")
    rp = resumen_prov.copy().rename(columns={
        "provincia": "Provincia", "region": "Región", "casos": "Casos",
        "edad_media": "Edad media", "prop_masc": "% Masculino"})
    st.dataframe(rp, width="stretch", hide_index=True, height=420)


# ============================== TAB 4: MODELO ==============================
with tab_modelo:
    st.subheader("¿Por qué SMOTE? Comparación de modelos")
    st.caption("Recall = capacidad de detectar los meses de alerta. Sin balanceo de clases, los modelos "
               "casi no detectan alertas pese a una exactitud alta.")

    met = metricas.copy()
    met["Balanceo"] = met["usa_smote"].map({True: "Con SMOTE", False: "Sin SMOTE"})
    colR, colF = st.columns(2)
    with colR:
        figR = px.bar(met, x="recall", y="modelo", color="Balanceo", barmode="group",
                      orientation="h", labels={"recall": "Recall", "modelo": ""},
                      color_discrete_map={"Con SMOTE": "#2A9D8F", "Sin SMOTE": "#C7CDD4"})
        figR.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), title="Recall")
        st.plotly_chart(figR, width="stretch")
    with colF:
        figF = px.bar(met, x="f1", y="modelo", color="Balanceo", barmode="group",
                      orientation="h", labels={"f1": "F1-Score", "modelo": ""},
                      color_discrete_map={"Con SMOTE": "#2A9D8F", "Sin SMOTE": "#C7CDD4"})
        figF.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), title="F1-Score")
        st.plotly_chart(figF, width="stretch")

    st.subheader("Tabla comparativa")
    tabla_met = met[["modelo", "Balanceo", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    tabla_met = tabla_met.rename(columns={
        "modelo": "Modelo", "accuracy": "Accuracy", "precision": "Precisión",
        "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC", "pr_auc": "PR-AUC"})

    # Degradé verde en la columna F1, sin depender de matplotlib
    def resaltar_f1(col):
        vmin, vmax = col.min(), col.max()
        estilos = []
        for v in col:
            t = 0 if vmax == vmin else (v - vmin) / (vmax - vmin)
            r = int(round(255 + t * (46 - 255)))
            g = int(round(255 + t * (157 - 255)))
            b = int(round(255 + t * (143 - 255)))
            txt = "white" if t > 0.6 else "black"
            estilos.append(f"background-color: rgb({r},{g},{b}); color: {txt};")
        return estilos

    st.dataframe(
        tabla_met.style.format({c: "{:.3f}" for c in ["Accuracy", "Precisión", "Recall", "F1", "ROC-AUC", "PR-AUC"]})
                       .apply(resaltar_f1, subset=["F1"]),
        width="stretch", hide_index=True)

    st.subheader("Importancia de variables (modelo seleccionado)")
    imp = importancia.sort_values("importancia", ascending=True)
    figI = px.bar(imp, x="importancia", y="variable", orientation="h",
                  labels={"importancia": "Importancia", "variable": ""})
    figI.update_traces(marker_color=AZUL)
    figI.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figI, width="stretch")


# ----------------------------------------------------------------------------
# Nota de uso responsable (pie de página)
# ----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="nota"><b>Uso responsable.</b> Este tablero presenta estimaciones estadísticas '
    'agregadas con fines académicos y de salud pública; no identifica personas ni constituye un '
    'diagnóstico individual. El suicidio es prevenible. Si vos o alguien que conocés atraviesa una '
    'crisis, en Argentina podés comunicarte con el Centro de Asistencia al Suicida (CAS) al '
    '<b>(011) 5275-1135</b> o al <b>0800-345-1435</b>, todos los días.</div>',
    unsafe_allow_html=True)
st.caption("Universidad Nacional de Salta · Licenciatura en Análisis de Sistemas · "
           "Metodos Cuantitativos para la Toma de Decisiones · 2025")
