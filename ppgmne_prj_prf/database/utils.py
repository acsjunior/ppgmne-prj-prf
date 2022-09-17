import ssl
import urllib.request as request
from io import BytesIO
from typing import Literal, Union
from zipfile import ZipFile

import pandas as pd
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
