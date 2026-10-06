import geopandas as gpd


def read_sites(path: str, layer: str, id_field: str) -> gpd.GeoDataFrame:
    """Read site points, index them by `id_field`."""
    gdf = gpd.read_file(path, layer=layer)
    gdf = gdf[[id_field, "geometry"]].rename(columns={id_field: "site"})
    if gdf["site"].duplicated().any():
        raise ValueError(f"duplicate site ids in field {id_field!r}")
    return gdf.set_index("site")
