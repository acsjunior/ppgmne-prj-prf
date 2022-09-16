import ssl
import urllib.request as request
from io import BytesIO
from zipfile import ZipFile

import pandas as pd


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
