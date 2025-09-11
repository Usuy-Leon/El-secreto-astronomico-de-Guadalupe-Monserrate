import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# The same points dictionary from your original script
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

EARTH_RADIUS_KM = 6371.0

def geodetic_to_ecef(lat_deg, lon_deg, alt_m=0):
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    alt_km = alt_m / 1000.0
    r = EARTH_RADIUS_KM + alt_km
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z])

# Create 3D plot
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
colors = ['#8B4000', '#556B2F', '#D2B48C', '#A0522D', '#6B8E23']

for i, (name, points) in enumerate(triangles.items()):
    # Convert GPS points to ECEF
    ecef_points = np.array([geodetic_to_ecef(*p) for p in points])
    
    # Plot vertices
    ax.scatter(ecef_points[:, 0], ecef_points[:, 1], ecef_points[:, 2], label=name, color=colors[i % len(colors)], s=50)
    
    # Draw triangle edges
    tri_edges = np.vstack([ecef_points, ecef_points[0]])  # Close the triangle loop
    ax.plot(tri_edges[:, 0], tri_edges[:, 1], tri_edges[:, 2], color=colors[i % len(colors)])
    
    # Draw filled triangle with some transparency
    verts = [list(ecef_points)]
    ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, facecolor=colors[i % len(colors)]))

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('3D Visualization of Triangles Formed by Historic Landmarks in Bogotá')
ax.legend()
plt.show()