import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from stl import mesh

def reproject_geotiff_to_utm(input_geotiff_path, output_geotiff_path, target_epsg_code):
    with rasterio.open(input_geotiff_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            CRS.from_epsg(target_epsg_code),
            src.width,
            src.height,
            *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': CRS.from_epsg(target_epsg_code),
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(output_geotiff_path, 'w', **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=CRS.from_epsg(target_epsg_code),
                resampling=Resampling.nearest
            )
    print(f"GeoTIFF reprojected and saved to: {output_geotiff_path}")

def raise_elevation_at_points(dem, transform, points, raise_height):

    for lon, lat in points:
        # Convert coordinates to raster indices
        col, row = ~transform * (lon, lat)
        row, col = int(round(row)), int(round(col))
        if 0 <= row < dem.shape[0] and 0 <= col < dem.shape[1]:
            # Raise a 3x3 neighborhood around the point for visibility
            r_min = max(row - 1, 0)
            r_max = min(row + 2, dem.shape[0])
            c_min = max(col - 1, 0)
            c_max = min(col + 2, dem.shape[1])
            dem[r_min:r_max, c_min:c_max] += raise_height
    return dem

def geotiff_to_stl(
    geotiff_path,
    stl_path,
    scale_xy=1.0,
    scale_z=1.0,
    base_height=0.1,
    utm_epsg=32618,
    highlight_points=None,
    raise_height=100.0
):
    with rasterio.open(geotiff_path) as src:
        dem = src.read(1)
        transform = src.transform
        nodata = src.nodatavals[0]

        if nodata is not None and not np.isnan(nodata):
            dem_valid = dem[dem != nodata]
            min_valid_dem = np.min(dem_valid) if dem_valid.size > 0 else 0
            dem = np.where(dem == nodata, min_valid_dem, dem)

        if highlight_points:
            dem = raise_elevation_at_points(dem, transform, highlight_points, raise_height)

        n_rows, n_cols = dem.shape
        x_coords = np.arange(n_cols) * transform.a + transform.c
        y_coords = np.arange(n_rows) * transform.e + transform.f
        y_coords = y_coords[::-1]  # Flip y to match DEM orientation

        # Apply horizontal scale (in millimeters per meter)
        x_coords = x_coords * scale_xy
        y_coords = y_coords * scale_xy

        xx, yy = np.meshgrid(x_coords, y_coords)
        zz = dem * scale_z

        # Create vertices and faces
        vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        faces = []
        for row in range(n_rows - 1):
            for col in range(n_cols - 1):
                v0 = row * n_cols + col
                v1 = v0 + 1
                v2 = v0 + n_cols
                v3 = v2 + 1
                faces.append([v0, v1, v2])
                faces.append([v1, v3, v2])
        faces = np.array(faces)

    # Create base for solid model
    min_z = np.min(zz)
    base_z = min_z - base_height

    num_vertices = vertices.shape[0]
    base_vertices = vertices.copy()
    base_vertices[:, 2] = base_z

    all_vertices = np.vstack([vertices, base_vertices])

    top_left = 0
    top_right = n_cols - 1
    bottom_left = (n_rows - 1) * n_cols
    bottom_right = n_rows * n_cols - 1

    base_top_left = top_left + num_vertices
    base_top_right = top_right + num_vertices
    base_bottom_left = bottom_left + num_vertices
    base_bottom_right = bottom_right + num_vertices

    base_faces = [
        [base_top_left, base_bottom_left, base_top_right],
        [base_top_right, base_bottom_left, base_bottom_right],
    ]

    wall_faces = []

    # Bottom edge walls
    for col in range(n_cols - 1):
        s0 = (n_rows - 1) * n_cols + col
        s1 = s0 + 1
        b0 = s0 + num_vertices
        b1 = s1 + num_vertices
        wall_faces.append([s0, s1, b1])
        wall_faces.append([s0, b1, b0])

    # Top edge walls
    for col in range(n_cols - 1):
        s0 = col
        s1 = col + 1
        b0 = s0 + num_vertices
        b1 = s1 + num_vertices
        wall_faces.append([s0, b1, s1])
        wall_faces.append([s0, b0, b1])

    # Left edge walls
    for row in range(n_rows - 1):
        s0 = row * n_cols
        s1 = (row + 1) * n_cols
        b0 = s0 + num_vertices
        b1 = s1 + num_vertices
        wall_faces.append([s0, s1, b1])
        wall_faces.append([s0, b1, b0])

    # Right edge walls
    for row in range(n_rows - 1):
        s0 = row * n_cols + (n_cols - 1)
        s1 = (row + 1) * n_cols + (n_cols - 1)
        b0 = s0 + num_vertices
        b1 = s1 + num_vertices
        wall_faces.append([s0, b1, s1])
        wall_faces.append([s0, b0, b1])

    all_faces = np.vstack([faces, base_faces, wall_faces])

    # Build and save STL mesh
    terrain_mesh = mesh.Mesh(np.zeros(all_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(all_faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = all_vertices[f[j], :]

    terrain_mesh.save(stl_path)
    print(f"STL file saved at: {stl_path}")

if __name__ == "__main__":
    raw_geotiff = "/Path//output.tif"
    reprojected_geotiff = "//Path/YourFiles/rasters_SRTMGL3/output.tif"
    output_stl = "/Path/output_solid.stl"

    utm_epsg_code = 32618

    # Step 1: Reproject GeoTIFF to UTM
    reproject_geotiff_to_utm(raw_geotiff, reprojected_geotiff, utm_epsg_code)

    # Step 2: Calculate scale to output model ~15 cm max side in millimeters
    desired_max_print_mm = 150  # 15 cm
    with rasterio.open(reprojected_geotiff) as src:
        real_width_m = src.width * abs(src.transform.a)
        real_height_m = src.height * abs(src.transform.e)
        max_dim_m = max(real_width_m, real_height_m)

    calculated_scale_xy = desired_max_print_mm / max_dim_m  # mm per meter
    print(f"Calculated horizontal scale (mm/m): {calculated_scale_xy}")

    vertical_factor = 2.0  # vertical exaggeration factor
    calculated_scale_z = calculated_scale_xy * vertical_factor
    print(f"Calculated vertical scale (mm/m): {calculated_scale_z}")

    # Base height in DEM units (meters). Calculate so that base is ~2 mm thick after scaling.
    desired_base_thickness_mm = 1.0
    calculated_base_height = desired_base_thickness_mm / calculated_scale_z
    print(f"Calculated base height (DEM units): {calculated_base_height}")


    # Raise height: bump height in DEM units for labels (~5 mm bump)
    desired_bump_mm = 100.0
    raise_height_dem_units = desired_bump_mm / calculated_scale_z
    print(f"Calculated raise height (DEM units): {raise_height_dem_units}")

    # Step 3: Generate STL
    geotiff_to_stl(
        reprojected_geotiff,
        output_stl,
        scale_xy=calculated_scale_xy,
        scale_z=calculated_scale_z,
        base_height=calculated_base_height,
        utm_epsg=utm_epsg_code,
        highlight_points=points_of_interest,
        raise_height=raise_height_dem_units
    )




