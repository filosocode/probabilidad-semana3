import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


# Funciones
def f(x):
    return 100  # trayectoria del dron constante


def g1(x):
    return 0.2 * x + 60  # terreno en [0,100]


def g2(x):
    return -0.1 * x + 90  # terreno en (100,200]


# Cálculo de áreas
A1, _ = quad(lambda x: f(x) - g1(x), 0, 100)
A2, _ = quad(lambda x: f(x) - g2(x), 100, 200)
A_total = A1 + A2

# Datos para graficar
x1 = np.linspace(0, 100, 200)
x2 = np.linspace(100, 200, 200)
y_f1 = np.full_like(x1, 100)
y_f2 = np.full_like(x2, 100)
y_g1 = g1(x1)
y_g2 = g2(x2)

# Gráfico
plt.figure(figsize=(10, 6))
# Trayectoria del dron
plt.plot(x1, y_f1, "r", linewidth=2)
plt.plot(x2, y_f2, "r", linewidth=2, label="Trayectoria del dron f(x)=100")
# Terreno
plt.plot(x1, y_g1, "b", linewidth=2)
plt.plot(x2, y_g2, "b", linewidth=2, label="Terreno g(x)")

# Área 1
plt.fill_between(x1, y_f1, y_g1, where=(y_f1 > y_g1), color="skyblue", alpha=0.5)
plt.text(
    30,
    80,
    f"A1 ≈ {A1:.0f} m²",
    fontsize=11,
    color="black",
    bbox=dict(facecolor="white", alpha=0.7),
)

# Área 2
plt.fill_between(x2, y_f2, y_g2, where=(y_f2 > y_g2), color="lightgreen", alpha=0.5)
plt.text(
    140,
    80,
    f"A2 ≈ {A2:.0f} m²",
    fontsize=11,
    color="black",
    bbox=dict(facecolor="white", alpha=0.7),
)

# Resultado total
plt.text(
    80,
    70,
    f"Área total ≈ {A_total:.0f} m²",
    fontsize=13,
    color="black",
    bbox=dict(facecolor="yellow", alpha=0.8),
)

# Estilo
plt.title("Áreas de seguridad entre dron y terreno")
plt.xlabel("Distancia en el eje x (m)")
plt.ylabel("Altura (m)")
plt.legend()
plt.grid(True)
plt.show()

print(f"A1 ≈ {A1:.0f} m², A2 ≈ {A2:.0f} m², Área total ≈ {A_total:.0f} m²")
