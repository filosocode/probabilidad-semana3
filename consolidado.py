# ============================================
# Proyecto de Probabilidad – Consolidado Semanas 3, 4 y 5
# Reporte ÚNICO en PDF con todos los gráficos y cálculos
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sns.set_theme(style="whitegrid")

print("=== INICIO DEL PROCESO ===")

# === Leer datos ===
df = pd.read_excel("data/Datos.xlsx")
df["Municipio"] = df["Municipio"].str.strip().str.upper()
datos_filtrado = df.copy()
print(f"Municipios incluidos: {datos_filtrado['Municipio'].unique()}")


# === Funciones auxiliares ===
def cv(serie):
    return serie.std() / serie.mean()


def calcular_weibull(mean, sd):
    k = (sd / mean) ** (-1.086)
    c = mean / gamma(1 + 1 / k)
    vmp = c * ((k - 1) / k) ** (1 / k) if k > 1 else None
    vmaxE = c * ((k + 2) / k) ** (1 / k)
    return k, c, vmp, vmaxE


def F_weibull(v, k, c):
    return 1 - np.exp(-((v / c) ** k))


# === Estadísticos generales ===
cv_resultados = datos_filtrado.groupby("Municipio").agg(
    media_viento=("vel_viento (m/s)", "mean"),
    sd_viento=("vel_viento (m/s)", "std"),
    CV_viento=("vel_viento (m/s)", cv),
    media_temp=("T (°C)", "mean"),
    sd_temp=("T (°C)", "std"),
    CV_temp=("T (°C)", cv),
)

weibull_params = []
for mun, row in cv_resultados.iterrows():
    k, c, vmp, vmaxE = calcular_weibull(row["media_viento"], row["sd_viento"])
    weibull_params.append(
        [mun, row["media_viento"], row["sd_viento"], row["CV_viento"], k, c, vmp, vmaxE]
    )

weibull_df = pd.DataFrame(
    weibull_params,
    columns=[
        "Municipio",
        "Media Viento",
        "SD Viento",
        "CV Viento",
        "k",
        "c",
        "vmp",
        "vmaxE",
    ],
)

# Cuartiles y probabilidades (Semana 5)
q_stats = datos_filtrado.groupby("Municipio").agg(
    Q1=("vel_viento (m/s)", lambda x: np.percentile(x, 25)),
    Q3=("vel_viento (m/s)", lambda x: np.percentile(x, 75)),
)
q_stats["RIC"] = q_stats["Q3"] - q_stats["Q1"]

prob_q = []
prob_p60 = []
for _, row in weibull_df.iterrows():
    mun, k, c = row["Municipio"], row["k"], row["c"]
    Q1, Q3 = q_stats.loc[mun, ["Q1", "Q3"]]
    prob = F_weibull(Q3, k, c) - F_weibull(Q1, k, c)
    prob_q.append([mun, prob])
    v60 = c * (-np.log(1 - 0.60)) ** (1 / k)
    prob60 = 1 - F_weibull(v60, k, c)
    prob_p60.append([mun, v60, prob60])

prob_q_df = pd.DataFrame(prob_q, columns=["Municipio", "Prob(Q1-Q3)"])
prob_p60_df = pd.DataFrame(prob_p60, columns=["Municipio", "v60", "Prob>v60"])

# === Consolidado en un solo PDF ===
with PdfPages("Consolidado.pdf") as pdf:
    # --- Semana 3 ---
    for var, titulo in [
        ("vel_viento (m/s)", "Velocidad del Viento"),
        ("T (°C)", "Temperatura"),
    ]:
        for mun in datos_filtrado["Municipio"].unique():
            plt.figure(figsize=(8, 5))
            sns.histplot(
                data=datos_filtrado[datos_filtrado["Municipio"] == mun],
                x=var,
                bins=30,
                alpha=0.6,
            )
            plt.title(f"Histograma de {titulo} – {mun}")
            pdf.savefig()
            plt.close()
        plt.figure(figsize=(8, 5))
        sns.histplot(data=datos_filtrado, x=var, hue="Municipio", bins=30, alpha=0.6)
        plt.title(f"Histograma comparativo de {titulo}")
        plt.xticks(rotation=90, ha="center")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    for var, titulo in [
        ("vel_viento (m/s)", "Velocidad del Viento"),
        ("T (°C)", "Temperatura"),
    ]:
        plt.figure(figsize=(8, 6))
        sns.boxplot(
            data=datos_filtrado, x="Municipio", y=var, palette="Set2", showfliers=False
        )
        plt.title(f"Boxplot de {titulo}")
        plt.xticks(rotation=90, ha="center")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Tablas
    for df_table, titulo in [
        (cv_resultados.round(3).reset_index(), "Coeficiente de Variación"),
        (weibull_df.round(3).reset_index(), "Parámetros Weibull"),
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
        tabla.scale(1, 1)
        pdf.savefig()
        plt.close()

    # --- Semana 4 ---
    for _, row in weibull_df.iterrows():
        mun, k, c = row["Municipio"], row["k"], row["c"]
        data = datos_filtrado[datos_filtrado["Municipio"] == mun]["vel_viento (m/s)"]
        v = np.linspace(0, data.max() + 5, 200)
        f_v = (k / c) * (v / c) ** (k - 1) * np.exp(-((v / c) ** k))
        plt.figure(figsize=(8, 5))
        sns.histplot(data, bins=30, stat="density", alpha=0.6, color="skyblue")
        plt.plot(v, f_v, "r-", lw=2, label=f"Weibull (k={k:.2f}, c={c:.2f})")
        plt.title(f"Histograma vs Weibull – {mun}")
        plt.xlabel("Velocidad del viento (m/s)")
        plt.ylabel("Densidad de probabilidad")
        plt.legend()
        pdf.savefig()
        plt.close()

    # --- Semana 5 ---
    for df_table, titulo in [
        (q_stats.round(3).reset_index(), "Cuartiles y RIC"),
        (prob_q_df.round(4), "Probabilidad entre Q1 y Q3"),
        (prob_p60_df.round(4), "Probabilidad de superar Percentil 60"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 3))
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
        tabla.scale(1, 1)
        pdf.savefig()
        plt.close()

    # Gráficos de RIC y probabilidades
    plt.figure(figsize=(10, 6))
    sns.barplot(data=q_stats.reset_index(), x="Municipio", y="RIC", palette="Set2")
    plt.title("Comparación del RIC")
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=prob_q_df, x="Municipio", y="Prob(Q1-Q3)", palette="Set2")
    plt.title("Probabilidad entre Q1 y Q3")
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=prob_p60_df, x="Municipio", y="Prob>v60", palette="Set2")
    plt.title("Probabilidad de superar Percentil 60")
    plt.xticks(rotation=90, ha="center")
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Curvas Weibull con Q1, Q3 y P60
    for _, row in weibull_df.iterrows():
        mun, k, c = row["Municipio"], row["k"], row["c"]
        data = datos_filtrado[datos_filtrado["Municipio"] == mun]["vel_viento (m/s)"]
        v = np.linspace(0, data.max() + 5, 300)
        f_v = (k / c) * (v / c) ** (k - 1) * np.exp(-((v / c) ** k))
        Q1, Q3 = q_stats.loc[mun, ["Q1", "Q3"]]
        v60 = prob_p60_df.loc[prob_p60_df["Municipio"] == mun, "v60"].values[0]
        plt.figure(figsize=(8, 5))
        sns.histplot(data, bins=30, stat="density", alpha=0.6, color="skyblue")
        plt.plot(v, f_v, "r-", lw=2, label=f"Weibull (k={k:.2f}, c={c:.2f})")
        plt.axvline(Q1, color="g", linestyle="--", label="Q1")
        plt.axvline(Q3, color="b", linestyle="--", label="Q3")
        plt.axvline(v60, color="purple", linestyle="--", label="P60")
        plt.title(f"Weibull con Q1, Q3 y P60 – {mun}")
        plt.xlabel("Velocidad del viento (m/s)")
        plt.ylabel("Densidad de probabilidad")
        plt.legend()
        pdf.savefig()
        plt.close()

print("=== REPORTE GENERADO: Consolidado.pdf ===")
