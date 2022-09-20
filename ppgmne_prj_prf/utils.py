import ssl
import urllib.request as request
from io import BytesIO
from typing import List, Literal, Union
from zipfile import ZipFile

import numpy as np
import pandas as pd
from haversine import haversine_vector
from loguru import logger
from unidecode import unidecode


def csv_zip_to_df(
    url: str,
    file_name: str,
    sep: str = ";",
    dec: str = ".",
    encoding: str = "ISO=8859-1",
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    url : str
        URL para download
    file_name : str
        Nome do arquivo para descompactar
    sep : str, optional
        Delimitador do csv, by default ";"
    dec : str, optional
        Separador decimal, by default "."
    encoding : str, optional
        Encoding, by default "ISO=8859-1"

    Returns
    -------
    pd.DataFrame
        Data frame descompactado
    """

    context = ssl._create_unverified_context()
    r = request.urlopen(url, context=context).read()
    csv_file = ZipFile(BytesIO(r)).open(file_name)

    df = pd.read_csv(csv_file, delimiter=sep, decimal=dec, encoding=encoding)

    return df


def clean_string(
    df: pd.DataFrame,
    target: Union[list, str],
    mode: Literal["lower", "upper"] = "lower",
) -> pd.DataFrame:
    """_summary_

    Parameters
    ----------
    df : pd.DataFrame
        Data frame completo
    target : Union[list, str]
        Campo ou lista de campos para limpeza
    mode : Literal["lower", "upper"], optional
        Modo de conversão dos caracteres, by default "lower"

    Returns
    -------
    pd.DataFrame
        Data frame tratado
    """
    if isinstance(target, str):
        target = [target]
    for col in target:
        df[col] = df[col].apply(unidecode).str.strip().str.lower()
        if mode == "upper":
            df[col] = df[col].str.upper()
    return df


def get_decimal_places(col: pd.Series) -> pd.Series:
    """_summary_

    Parameters
    ----------
    col : pd.Series
        Coluna numérica para contagem de casas decimais

    Returns
    -------
    pd.Series
        Vetor com a contagem de casas decimais de cada elemento
    """
    out = col.astype(float).astype(str).str.split(".").str[1].str.len().astype(int)
    return out


def concatenate_dict_of_dicts(dict_of_dicts: dict) -> dict:
    """_summary_

    Parameters
    ----------
    dict_of_dicts : dict
        Dicionário de dicionários com a mesma estrutura.

    Returns
    -------
    dict
        Dictionário unificado.
    """
    dict_out = None

    for key in dict_of_dicts:
        dict_i = dict_of_dicts[key]

        if dict_out is None:
            dict_out = dict_i
        else:
            dict_out = dict(dict_out, **dict_i)

    return dict_out


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


def trace_df(df: pd.DataFrame) -> pd.DataFrame:
    """Imprime as dimensões do data frame.

    Parameters
    ----------
    df : pd.DataFrame
        Base de dados.

    Returns
    -------
    pd.DataFrame
        Base de dados.
    """
    logger.info(f"shape: {df.shape}")
    return df
