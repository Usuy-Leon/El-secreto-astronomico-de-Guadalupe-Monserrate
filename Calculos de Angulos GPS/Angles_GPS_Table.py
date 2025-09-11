import math

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in kilometers
    r = 6371

    # Calculate the result
    distance = c * r

    return distance

def main():
    # GPS coordinates (latitude, longitude, elevation)
    # Point A: (4.3530, -74.0315, 3265)
    # Point B: (4.3620, -74.0319, 3161)

    # Extracting latitude and longitude (ignoring elevation as specified)
    point_a = (4.3530, -74.0315)
    point_b = (4.3620, -74.0319)

    lat_a, lon_a = point_a
    lat_b, lon_b = point_b

    # Calculate distance
    distance = haversine_distance(lat_a, lon_a, lat_b, lon_b)

    # Display results
    print("GPS Distance Calculator")
    print("=" * 30)
    print(f"Point A: Latitude = {lat_a}°, Longitude = {lon_a}°")
    print(f"Point B: Latitude = {lat_b}°, Longitude = {lon_b}°")
    print(f"Distance between points: {distance:.4f} km")
    print(f"Distance between points: {distance * 1000:.2f} meters")

    # Additional information
    print("\nNote: Calculation assumes both points are at the same height")
    print("Formula used: Haversine formula for great-circle distance")

if __name__ == "__main__":
    main()
