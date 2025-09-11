import math
import numpy as np
from scipy.optimize import minimize

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

def ecef_to_geodetic(x, y, z):
    r = np.linalg.norm([x, y, z])
    lat = math.asin(z / r)
    lon = math.atan2(y, x)
    alt_km = r - EARTH_RADIUS_KM
    return (math.degrees(lat), math.degrees(lon), alt_km * 1000)

def angular_distance(vec1, vec2):
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.acos(cos_angle)

# Fixed points A and B
A_lat, A_lon, A_alt = 4.591840782861174, -74.05436430808176, 3265
B_lat, B_lon, B_alt = 4.605591310552596, -74.0555291341563, 3161
C_alt_m = 2630  # fixed altitude in meters

# Given triangle angles
angle_A = math.radians(67.1)
angle_B = math.radians(67.2)
angle_C = math.radians(45.7)

# Convert fixed points to ECEF
A_ecef = geodetic_to_ecef(A_lat, A_lon, A_alt)
B_ecef = geodetic_to_ecef(B_lat, B_lon, B_alt)

# Calculate side c (distance between A and B)
side_c = angular_distance(A_ecef, B_ecef)

# Calculate sides a and b using spherical law of sines
side_a = side_c * math.sin(angle_A) / math.sin(angle_C)
side_b = side_c * math.sin(angle_B) / math.sin(angle_C)

R = EARTH_RADIUS_KM + C_alt_m / 1000.0  # sphere radius at altitude of C in km

# Objective function to minimize:
# Sum of squared differences between desired and actual angular distances to A and B for point C
def objective(coord):
    lat_deg, lon_deg = coord
    C_ecef = geodetic_to_ecef(lat_deg, lon_deg, C_alt_m)
    dist_to_A = angular_distance(C_ecef, A_ecef)
    dist_to_B = angular_distance(C_ecef, B_ecef)
    return (dist_to_A - side_b)**2 + (dist_to_B - side_a)**2

# Constraint: C lies on sphere surface of radius R (enforced by fixing altitude)
# Here, altitude fixed by geodetic_to_ecef, so no extra constraint needed

# Initial guess for C - midpoint between A and B
init_lat = (A_lat + B_lat) / 2
init_lon = (A_lon + B_lon) / 2
initial_guess = [init_lat, init_lon]

# Run optimization twice with offset initial guesses to get both possible solutions
result1 = minimize(objective, initial_guess, method='Nelder-Mead')
result2 = minimize(objective, [initial_guess[0], initial_guess[1] + 0.01], method='Nelder-Mead')

C1_lat, C1_lon = result1.x
C2_lat, C2_lon = result2.x

C1_ecef = geodetic_to_ecef(C1_lat, C1_lon, C_alt_m)
C2_ecef = geodetic_to_ecef(C2_lat, C2_lon, C_alt_m)

print("Possible solutions for point C coordinates with fixed altitude:")
print(f"Solution 1: Latitude: {C1_lat:.8f}°, Longitude: {C1_lon:.8f}°, Altitude: {C_alt_m} m")
print(f"Distance to A: {math.degrees(angular_distance(C1_ecef, A_ecef)):.4f}°, Distance to B: {math.degrees(angular_distance(C1_ecef, B_ecef)):.4f}°")

print(f"Solution 2: Latitude: {C2_lat:.8f}°, Longitude: {C2_lon:.8f}°, Altitude: {C_alt_m} m")
print(f"Distance to A: {math.degrees(angular_distance(C2_ecef, A_ecef)):.4f}°, Distance to B: {math.degrees(angular_distance(C2_ecef, B_ecef)):.4f}°")








'''
import math
import numpy as np

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

def ecef_to_geodetic(x, y, z):
    r = np.linalg.norm([x, y, z])
    lat = math.asin(z / r)
    lon = math.atan2(y, x)
    alt_km = r - EARTH_RADIUS_KM
    return (math.degrees(lat), math.degrees(lon), alt_km * 1000)

def angular_distance(vec1, vec2):
    """Returns central angle in radians between two ECEF vectors."""
    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return math.acos(cos_angle)

def chord_length(angle_rad, radius):
    """Calculate chord length from central angle and radius."""
    return 2 * radius * math.sin(angle_rad / 2)

# Fixed points A and B with altitude
A_lat, A_lon, A_alt = 4.591840782861174, -74.05436430808176, 3265
B_lat, B_lon, B_alt = 4.605591310552596, -74.0555291341563, 3161
C_alt = 2630

# Convert A and B to ECEF
A_ecef = geodetic_to_ecef(A_lat, A_lon, A_alt)
B_ecef = geodetic_to_ecef(B_lat, B_lon, B_alt)

# Calculate angular distance (side c in radians) between A and B
side_c = angular_distance(A_ecef, B_ecef)

# Given triangle angles in radians
angle_A = math.radians(67.1)
angle_B = math.radians(67.2)
angle_C = math.radians(45.7)

# Use law of sines for spherical triangle to find sides a and b
side_a = side_c * math.sin(angle_A) / math.sin(angle_C)
side_b = side_c * math.sin(angle_B) / math.sin(angle_C)

# Define coordinate system with A at origin and AB as x-axis unit vector
AB = B_ecef - A_ecef
AB_unit = AB / np.linalg.norm(AB)

# Define a vector 'up' to find a plane normal, avoiding parallelism
up = np.array([0, 0, 1])
if np.allclose(AB_unit, up) or np.allclose(AB_unit, -up):
    up = np.array([0, 1, 0])

normal = np.cross(AB_unit, up)
normal /= np.linalg.norm(normal)

# Third axis orthogonal to AB_unit and normal (not used directly here)
third_axis = np.cross(normal, AB_unit)

# Sphere radius including altitude of point C
R = EARTH_RADIUS_KM + C_alt / 1000

# Calculate chord lengths on sphere of radius R
chord_c = np.linalg.norm(B_ecef - A_ecef)
chord_a = chord_length(side_a, R)
chord_b = chord_length(side_b, R)

# Distance between centers
d = chord_c
r0 = chord_b
r1 = chord_a

# Check feasibility of intersection
if d > r0 + r1 or d < abs(r0 - r1):
    print("No valid solution for point C with given distances")
else:
    # Compute intersection circle center point P2 on line A-B
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = math.sqrt(abs(r0**2 - a**2))

    P0 = A_ecef
    P1 = B_ecef

    P2 = P0 + a * (P1 - P0) / d

    # Two intersection points (solutions) for C
    inter1 = P2 + h * normal
    inter2 = P2 - h * normal

    # Convert both to geodetic coordinates
    C1 = ecef_to_geodetic(*inter1)
    C2 = ecef_to_geodetic(*inter2)

    print("Possible solutions for point C coordinates:")
    print(f"Solution 1: Latitude: {C1[0]:.8f}°, Longitude: {C1[1]:.8f}°, Altitude: {C1[2]:.1f} m")
    print(f"Solution 2: Latitude: {C2[0]:.8f}°, Longitude: {C2[1]:.8f}°, Altitude: {C2[2]:.1f} m")
    
    '''

