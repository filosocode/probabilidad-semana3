# Proyecto de Probabilidad – Semanas 3, 4 y 5

Este repositorio contiene el desarrollo de las **actividades 1, 2 y 3** del curso de Probabilidad y Estadística, aplicadas al análisis de datos meteorológicos de **Barranquilla** y **Medellín**.

## Descripción

El proyecto sigue la metodología propuesta en el material académico:contentReference[oaicite:3]{index=3}y en artículos de referencia sobre la distribución de Weibull:contentReference[oaicite:4]{index=4}.

### Semana 3 – Actividad 1

- Histogramas y boxplots de velocidad del viento y temperatura.
- Cálculo de medias, desviación estándar y coeficiente de variación.
- Estimación de parámetros de la distribución de **Weibull** (k, c).
- Obtención de velocidades relevantes: **vmp** (más probable) y **vmaxE** (máxima energía).

### Semana 4 – Actividad 2

- Superposición de histogramas con curvas de distribución de Weibull.
- Comparación visual del ajuste y análisis del potencial eólico.
- Conclusiones comparativas entre Barranquilla y Medellín.

### Semana 5 – Actividad 3

- Cálculo de **cuartiles (Q1, Q3)** y del **rango intercuartílico (RIC)**.
- Probabilidad de que la velocidad esté entre Q1 y Q3.
- Probabilidad de superar el **percentil 60 (P60)**.
- Representación gráfica de probabilidades y conclusiones sobre el recurso eólico.

## Requisitos

- Python 3.9+
- Librerías: `pandas`, `matplotlib`, `seaborn`, `scipy`, `numpy`

Instalación rápida:

```bash
pip install -r requirements.txt
Ejecución
```

Colocar el archivo de datos Datos.xlsx en la carpeta data/ y ejecutar:

python semana3.py
python semana4.py
python semana5.py
Salidas
Reporte_Semana3.pdf: Histogramas, boxplots, coeficiente de variación y parámetros Weibull.

Reporte_Semana4.pdf: Ajuste de Weibull frente a histogramas y conclusiones comparativas.

Reporte_Semana5.pdf: Cuartiles, RIC, probabilidades (Q1-Q3, >P60) y gráficas complementarias.
