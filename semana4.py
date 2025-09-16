# ============================================
# Semana 3 – Actividad 1
# Municipios: Barranquilla y Medellín
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.stats import weibull_min

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
# Paso 4: Reporte Semana 3
# ============================================
print("Generando reporte de Semana 3...")
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

# ============================================
# Semana 4 – Actividad 2
# ============================================
print("Generando reporte de Semana 4...")

with PdfPages("Reporte_Semana4.pdf") as pdf:
    # Comparación Histograma vs Weibull
    # Comparación Histograma vs Weibull
    for _, row in weibull_df.iterrows():
        mun = row["Municipio"]  # 👈 ahora sí el nombre correcto
        data = datos_filtrado[datos_filtrado["Municipio"] == mun]["vel_viento (m/s)"]

        v = np.linspace(0, data.max() + 5, 200)
        k, c = row["k"], row["c"]
        f_v = (k / c) * (v / c) ** (k - 1) * np.exp(-((v / c) ** k))

        plt.figure(figsize=(8, 5))
        sns.histplot(data, bins=30, stat="density", alpha=0.6, color="skyblue")
        plt.plot(v, f_v, "r-", lw=2, label=f"Weibull (k={k:.2f}, c={c:.2f})")
        plt.title(f"Histograma vs Distribución Weibull – {mun}")
        plt.xlabel("Velocidad del viento (m/s)")
        plt.ylabel("Densidad de probabilidad")
        plt.legend()
        pdf.savefig()
        plt.close()

    # Texto explicativo
    # Conclusiones comparativas más detalladas
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axis("off")
    texto = (
        "Conclusiones – Semana 4\n\n"
        "1. Al superponer la curva de Weibull con los histogramas, se observa que el ajuste es adecuado "
        "en ambas ciudades, lo que confirma que esta distribución modela correctamente la variabilidad del viento.\n\n"
        "2. En Barranquilla, los parámetros Weibull (k≈2.78, c≈14.03) reflejan una mayor dispersión "
        "y velocidades medias más altas. Esto se traduce en valores de vmp≈11.96 m/s y vmaxE≈17.04 m/s, "
        "indicando un recurso eólico con mayor potencial energético.\n\n"
        "3. En Medellín, los parámetros (k≈4.11, c≈8.95) muestran una curva más concentrada en torno a la media. "
        "Las velocidades típicas son menores (vmp≈8.37 m/s y vmaxE≈9.86 m/s), lo cual limita el aprovechamiento "
        "para proyectos de gran escala.\n\n"
        "4. Comparando ambas ciudades, Barranquilla presenta un perfil más favorable para la generación de "
        "energía eólica, dado que combina mayor velocidad del viento con una adecuada dispersión, mientras "
        "que Medellín, aunque presenta regularidad, su rango de velocidades es más bajo.\n\n"
        "Conclusión general: Barranquilla es la ciudad con mayor probabilidad de generar energía eólica "
        "de manera eficiente en comparación con Medellín."
    )
    # Usamos bbox para darle márgenes y justificar
    ax.text(
        0.5,
        1,
        texto,
        va="top",
        ha="center",
        fontsize=10,
        wrap=True,
        bbox=dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.5"),
    )
    pdf.savefig()
    plt.close()


print("=== REPORTE GENERADO: Reporte_Semana4.pdf ===")
