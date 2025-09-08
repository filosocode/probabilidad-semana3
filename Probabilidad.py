# ============================================
# Semana 3 – Actividad 1
# Municipios: Barranquilla y Medellín
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma
from matplotlib.backends.backend_pdf import PdfPages

# Configuración general de estilo
sns.set_theme(style="whitegrid")

print("=== INICIO DEL PROCESO ===")

# ============================================
# Paso 1: Leer datos y normalizar nombres
# ============================================
print("Leyendo datos...")
df = pd.read_excel("data/Datos.xlsx")
df["Municipio"] = df["Municipio"].str.strip().str.upper()
datos_filtrado = df[df["Municipio"].isin(["BARRANQUILLA", "MEDELLÍN"])]
print(
    f"Datos filtrados para municipios seleccionados: {datos_filtrado['Municipio'].unique()}"
)


# ============================================
# Paso 2: Funciones auxiliares
# ============================================
def cv(serie):
    """Coeficiente de variación"""
    return serie.std() / serie.mean()


def calcular_weibull(mean, sd):
    """Devuelve parámetros k, c, vmp y vmaxE para Weibull"""
    k = (sd / mean) ** (-1.086)  # aproximación
    c = mean / gamma(1 + 1 / k)
    vmp = c * ((k - 1) / k) ** (1 / k) if k > 1 else None
    vmaxE = c * ((k + 2) / k) ** (1 / k)
    return k, c, vmp, vmaxE


# ============================================
# Paso 3: Estadísticos por municipio
# ============================================
print("Calculando estadísticos descriptivos y coeficiente de variación...")
cv_resultados = datos_filtrado.groupby("Municipio").agg(
    media_viento=("vel_viento (m/s)", "mean"),
    sd_viento=("vel_viento (m/s)", "std"),
    CV_viento=("vel_viento (m/s)", cv),
    media_temp=("T (°C)", "mean"),
    sd_temp=("T (°C)", "std"),
    CV_temp=("T (°C)", cv),
)
print(cv_resultados)

print("Calculando parámetros de la distribución Weibull...")
weibull_params = []
for mun, row in cv_resultados.iterrows():
    k, c, vmp, vmaxE = calcular_weibull(row["media_viento"], row["sd_viento"])
    weibull_params.append(
        [mun, row["media_viento"], row["sd_viento"], k, c, vmp, vmaxE]
    )

weibull_df = pd.DataFrame(
    weibull_params,
    columns=["Municipio", "Media Viento", "SD Viento", "k", "c", "vmp", "vmaxE"],
)
print(weibull_df)


# ============================================
# Paso 4: Reporte en PDF
# ============================================
print("Generando reporte en PDF...")
with PdfPages("Reporte_Semana3.pdf") as pdf:

    # Histogramas
    for var, titulo in [
        ("vel_viento (m/s)", "Velocidad del Viento"),
        ("T (°C)", "Temperatura"),
    ]:
        for mun in ["BARRANQUILLA", "MEDELLÍN"]:
            plt.figure(figsize=(8, 5))
            sns.histplot(
                data=datos_filtrado[datos_filtrado["Municipio"] == mun],
                x=var,
                bins=30,
                color="steelblue" if mun == "BARRANQUILLA" else "darkorange",
                alpha=0.6,
            )
            plt.title(f"Histograma de {titulo} – {mun}")
            pdf.savefig()
            plt.close()
        # comparativo
        plt.figure(figsize=(8, 5))
        sns.histplot(data=datos_filtrado, x=var, hue="Municipio", bins=30, alpha=0.6)
        plt.title(f"Histograma comparativo de {titulo}")
        pdf.savefig()
        plt.close()

    # Boxplots
    for var, titulo in [
        ("vel_viento (m/s)", "Velocidad del Viento"),
        ("T (°C)", "Temperatura"),
    ]:
        plt.figure(figsize=(6, 5))
        sns.boxplot(
            data=datos_filtrado,
            x="Municipio",
            y=var,
            hue="Municipio",
            palette="Set2",
            showfliers=False,
            legend=False,
        )
        plt.title(f"Boxplot de {titulo}")
        pdf.savefig()
        plt.close()

    # Texto explicativo
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    texto = (
        "Semana 3 – Actividad 1\n\n"
        "1. Se filtraron los municipios Barranquilla y Medellín.\n"
        "2. Se construyeron histogramas y boxplots para viento y temperatura.\n"
        "3. Se calcularon medias, desviaciones estándar y coeficientes de variación.\n"
        "4. Se estimaron los parámetros de la distribución Weibull (k, c).\n"
        "5. Se obtuvieron vmp y vmaxE para evaluar escenarios de energía eólica.\n"
    )
    ax.text(0, 1, texto, va="top", fontsize=10)
    pdf.savefig()
    plt.close()

    # Tablas
    cv_resultados_redondeado = cv_resultados.round(3).reset_index()
    weibull_df_redondeado = weibull_df.round(3).reset_index()

    for df_table, titulo in [
        (cv_resultados_redondeado, "Coeficiente de Variación"),
        (weibull_df_redondeado, "Parámetros Weibull"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.axis("off")
        ax.set_title(titulo, fontsize=12, pad=10, fontweight="bold")

        tabla = ax.table(
            cellText=df_table.values,
            colLabels=df_table.columns,
            loc="center",
            cellLoc="center",
        )

        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1.0, 1.0)

        for (row, col), cell in tabla.get_celld().items():
            cell.set_linewidth(0.5)
            cell.set_edgecolor("grey")
            if row == 0:
                cell.set_facecolor("#d9d9d9")
                cell.set_text_props(weight="bold", color="black")
            elif row % 2 == 0:
                cell.set_facecolor("#f2f2f2")

        pdf.savefig()
        plt.close()

print("=== REPORTE GENERADO: Reporte_Semana3.pdf ===")
