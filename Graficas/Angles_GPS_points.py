

"""
Este codigo calcula triangulos y sus angulos de la proyeccion solar del sol a puntos historicos
de La Candelaria, Bogotá, Colombia.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


# triángulos: cada punto es (latitud, longitud, altitud en metros)
triangles = {
    "Chorro de Quevedo": [
        (4.591840782861174, -74.05436430808176, 3265),
        (4.605591310552596, -74.0555291341563, 3161),
        (4.597090245765097, -74.06983744529906, 2633)
    ],
    "Catedral Primada": [
        (4.591840782861174, -74.05436430808176, 3265),
        (4.605591310552596, -74.0555291341563, 3161),
        (4.598204386347961, -74.0752537684042, 2607)
    ],
    "Iglesia la Candelaria": [
        (4.591840782861174, -74.05436430808176, 3265),
        (4.605591310552596, -74.0555291341563, 3161),
        (4.59669227968629, -74.07258806859734, 2624)
    ],
    "Iglesia Egipto": [
        (4.591840782861174, -74.05436430808176, 3265),
        (4.605591310552596, -74.0555291341563, 3161),
        (4.593619554868877, -74.06896952527157, 2590)
    ],
    "San Francisco": [
        (4.591840782861174, -74.05436430808176, 3265),
        (4.605591310552596, -74.0555291341563, 3161),
        (4.601791021450818, -74.07321911962411, 2605)
    ]
}

# Radio medio de la Tierra en kilómetros
EARTH_RADIUS_KM = 6371.0

def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0):
    """
    Convierte coordenadas geodésicas (latitud, longitud, altitud en metros)
    a coordenadas cartesianas ECEF (x, y, z) en kilómetros.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    alt_km = alt_m / 1000.0  
    r = EARTH_RADIUS_KM + alt_km
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z])

def calculate_angle_3d(A, B, C):
    """
    Calcula el ángulo en el punto B dado tres puntos A, B y C en coordenadas 3D.
    """
    BA = A - B
    BC = C - B
    cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

for name, points in triangles.items():
    # Convertir cada punto a coordenadas circulares 3D
    A = geodetic_to_ecef(*points[0])
    B = geodetic_to_ecef(*points[1])
    C = geodetic_to_ecef(*points[2])

    angle_A = calculate_angle_3d(B, A, C)
    angle_B = calculate_angle_3d(A, B, C)
    angle_C = calculate_angle_3d(A, C, B)

    print(f"| {name:<24} | {angle_A:10.2f}° | {angle_B:10.2f}° | {angle_C:10.2f}° |")



# Imprimir tabla con los ángulos calculados usando coordenadas 3D
print("| Triángulo                | Ángulo 1 (A) | Ángulo 2 (B) | Ángulo 3 (C) |")
print("|--------------------------|--------------|--------------|--------------|")


# Paleta de colores para graficar
terra_colors = [
    "#8B4000", "#556B2F", "#D2B48C", "#A0522D", "#6B8E23"
]

plt.figure(figsize=(12, 10))
ax = plt.gca()
ax.set_facecolor('white')

def draw_angle_arc(vertex, adj1, adj2, color, radius=0.0005):
    """
    Dibuja un arco pequeño (semicírculo) en el vértice dado,
    entre los dos puntos adyacentes adj1 y adj2.
    """
    v1 = adj1 - vertex
    v2 = adj2 - vertex
    ang1 = math.degrees(math.atan2(v1[1], v1[0])) % 360
    ang2 = math.degrees(math.atan2(v2[1], v2[0])) % 360
    start_angle = ang1
    end_angle = ang2
    if end_angle < start_angle:
        end_angle += 360
    arc_angle = end_angle - start_angle
    if arc_angle > 180:
        start_angle, end_angle = end_angle, start_angle + 360
    arc = Arc((vertex[0], vertex[1]), width=radius*2, height=radius*2,
              angle=0, theta1=start_angle, theta2=end_angle,
              color=color, lw=2)
    ax.add_patch(arc)

def angle_text_pos(vertex, adj1, adj2, offset=0.001, x_shift=0.0005):
    """
    Calcula la posición para el texto del ángulo, desplazándolo un poco a la derecha
    para evitar superposición con el semicírculo.
    """
    v1 = adj1 - vertex
    v2 = adj2 - vertex
    direction = v1 + v2
    norm = np.linalg.norm(direction)
    if norm == 0:
        pos = vertex
    else:
        direction = direction / norm
        pos = vertex + direction * offset
    pos[0] += x_shift  # Desplazar a la derecha
    return pos

for i, (name, points) in enumerate(triangles.items()):
    color = terra_colors[i % len(terra_colors)]

    # Extraer latitudes y longitudes para graficar (en orden cerrado)
    lats = [p[0] for p in points] + [points[0][0]]
    lons = [p[1] for p in points] + [points[0][1]]
    plt.plot(lons, lats, marker='o', linestyle='-', color=color, label=name)

    # Convertir a numpy arrays para cálculo de vectores en 2D (lon, lat)
    A = np.array([points[0][1], points[0][0]])
    B = np.array([points[1][1], points[1][0]])
    C = np.array([points[2][1], points[2][0]])

    # Calcular ángulo 3 usando coordenadas 3D para precisión
    angle_C = calculate_angle_3d(
        geodetic_to_ecef(*points[0]),
        geodetic_to_ecef(*points[2]),
        geodetic_to_ecef(*points[1])
    )

    # Dibujar arco pequeño en el vértice del ángulo 3
    draw_angle_arc(C, A, B, color, radius=0.0005)

    # Posicionar y dibujar texto del ángulo 3 desplazado a la derecha
    pos = angle_text_pos(C, A, B, offset=0.001, x_shift=0.0005)
    plt.text(pos[0], pos[1], f'{angle_C:.1f}°', fontsize=11, fontweight='bold',
             ha='center', va='center', color='black',
             bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))

plt.title('Ángulos de triángulos formados por iglesias de Bogotá (con curvatura y altitud)', color='black')
plt.xlabel('Longitud', color='black')
plt.ylabel('Latitud', color='black')
plt.grid(True, alpha=0.3, color='gray')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('/home/usuaryo/Downloads/triangles_with_arcs.png')
plt.show()
