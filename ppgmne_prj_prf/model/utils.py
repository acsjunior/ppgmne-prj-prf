from typing import List

import numpy as np
import pandas as pd
from haversine import haversine_vector
from loguru import logger


def get_distance_matrix(
    lat_rows: List[float],
    lon_rows: List[float],
    lat_cols: List[float] = None,
    lon_cols: List[float] = None,
    squared: bool = False,
) -> np.array:
    """_summary_

    Parameters
    ----------
    lat_rows : List[float]
        Lista com as latitudes verticais.
    lon_rows : List[float]
        Lista das longitudes verticais.
    lat_cols : List[float], optional
        Lista das latitudes horizontais, by default None
    lon_cols : List[float], optional
        Lista das longitudes horizontais, by default None
    squared : bool, optional
        Matriz quadrada. Se selecionado, lat_cols e lon_cols são ignorados, by default False

    Returns
    -------
    np.array
        Matriz de distâncias.
    """
    if squared:
        lat_cols = lat_rows
        lon_cols = lon_rows
        logger.info("Argumentos 'lat_cols' e 'lon_cols' ignorados.")

    coords_rows = np.array([x for x in zip(lat_rows, lon_rows)])
    coords_cols = np.array([x for x in zip(lat_cols, lon_cols)])

    dist_matrix = haversine_vector(coords_cols, coords_rows, unit="km", comb=True)

    return dist_matrix
