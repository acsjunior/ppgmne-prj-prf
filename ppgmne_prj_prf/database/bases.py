import json

import pandas as pd
import scipy.stats as stats
from loguru import logger
from pandera import Column, DataFrameSchema
from pandera.errors import SchemaError
from shapely.geometry import Point, Polygon

from ppgmne_prj_prf.config.params import (
    COORDS_MIN_DECIMAL_PLACES,
    STR_COLS_TO_LOWER,
    STR_COLS_TO_UPPER,
    UF,
)
from ppgmne_prj_prf.config.paths import (
    PATH_DATA_IBGE_BORDERS,
    PATH_DATA_PRF,
    PATH_DATA_PRF_CACHE,
)
from ppgmne_prj_prf.database.utils import (
    clean_string,
    csv_zip_to_df,
    get_decimal_places,
)


class Accidents:
    def __init__(self, verbose=True, read_cache=False):
        self.name_raw = "accidents_raw"
        self.name = "accidents"
        self.uf = UF
        self.verbose = verbose
        self.read_cache = read_cache

        logger.info("Lendo as urls dos acidentes.") if self.verbose else None
        with open(PATH_DATA_PRF / "accidents.json") as file:
            self.urls = json.load(file)

        self.key_in = "id"

        self.df_schema_in: dict[str, Column] = {
            "id": Column(int),
            "ano": Column(int),
            "data_inversa": Column(str),
            "dia_semana": Column(str),
            "horario": Column(str),
            "uf": Column(str),
            "municipio": Column(str),
            "causa_acidente": Column(str),
            "tipo_acidente": Column(str, nullable=True),
            "classificacao_acidente": Column(str),
            "fase_dia": Column(str),
            "sentido_via": Column(str),
            "tipo_pista": Column(str),
            "tracado_via": Column(str),
            "uso_solo": Column(str),
            "pessoas": Column(int),
            "mortos": Column(int),
            "feridos_leves": Column(int),
            "feridos_graves": Column(int),
            "ilesos": Column(int),
            "ignorados": Column(int),
            "feridos": Column(int),
            "veiculos": Column(int),
            "latitude": Column(str),
            "longitude": Column(str),
            "regional": Column(str, nullable=True),
            "delegacia": Column(str, nullable=True),
            "uop": Column(str, nullable=True),
        }

        self.key_out = "id"

        self.df_schema_out: dict[str, Column] = {
            "id": Column(int),
            "ano": Column(int),
            "data_hora": Column("datetime64[ns]"),
            "dia_semana": Column(str),
            "uf": Column(str),
            "municipio": Column(str),
            "causa_acidente": Column(str),
            "tipo_acidente": Column(str),
            "classificacao_acidente": Column(str),
            "fase_dia": Column(str),
            "sentido_via": Column(str),
            "tipo_pista": Column(str),
            "tracado_via": Column(str),
            "uso_solo": Column(str),
            "pessoas": Column(int),
            "mortos": Column(int),
            "feridos_leves": Column(int),
            "feridos_graves": Column(int),
            "ilesos": Column(int),
            "ignorados": Column(int),
            "feridos": Column(int),
            "veiculos": Column(int),
            "latitude": Column(float),
            "longitude": Column(float),
            "regional": Column(str),
            "delegacia": Column(str),
            "uop": Column(str),
        }

    def extract(self):
        logger.info(
            "Início da leitura dos registros de acidentes."
        ) if self.verbose else None

        cache_path = PATH_DATA_PRF_CACHE / f"{self.name_raw}.pkl"

        if self.read_cache:
            logger.info("Modo de leitura da cache ativo.")
            self.df_accidents = pd.read_pickle(cache_path)
            logger.info("Fim!")
            return

        df_out = pd.DataFrame()
        for year in self.urls.keys():

            # Lê os dados dos acidentes:
            url = self.urls[year]
            file_name = f"datatran{year}.csv"
            df = csv_zip_to_df(url, file_name)
            df["ano"] = year

            logger.info(
                f"Lendo os registros de acidentes de {year}."
            ) if self.verbose else None
            # Valida os dados de entrada:
            try:
                df = DataFrameSchema(
                    columns=self.df_schema_in,
                    unique=self.key_in,
                    coerce=True,
                    strict="filter",
                ).validate(df)
            except SchemaError as se:
                logger.error(f"Erro ao validar os dados dos acidentes de {year}.")
                logger.error(se)

            # Concatena os anos:
            if df_out.shape[0] == 0:
                df_out = df.copy()
            else:
                df_out = pd.concat([df_out, df], ignore_index=True)

        # Filtra a UF desejada:
        logger.info(
            f"Selecionando somente os dados do {self.uf}."
        ) if self.verbose else None
        df_out = df_out[df_out["uf"].str.upper() == self.uf].copy()

        # Armazena a cache caso o modo de leitura da cache não esteja ativo:
        if not self.read_cache:
            logger.info(f"Armazenado {cache_path}.")
            df_out.to_pickle(cache_path)

        self.df_accidents = df_out.copy()
        logger.info(f"Fim da extração dos dados.")

    def transform(self):
        logger.info("Início do pré processamento.") if self.verbose else None

        cache_path = PATH_DATA_PRF_CACHE / f"{self.name}.pkl"
        if self.read_cache:
            logger.info("Modo de leitura da cache ativo.")
            self.df_accidents = pd.read_pickle(cache_path)
            logger.info("Fim do pré processamento.")
            return

        df = self.df_accidents
        self.__print_df_shape(df)

        logger.info("Removendo os registros incompletos.") if self.verbose else None
        df.dropna(inplace=True)
        self.__print_df_shape(df)

        # Mantém na base somente registros nas delegacias das UFs desejadas:
        logger.info(
            f"Mantendo somente os registros do {self.uf}."
        ) if self.verbose else None
        df["delegacia"] = df["delegacia"].str.upper()
        df = df[
            df["delegacia"].str.contains("|".join([self.uf]))
        ].copy()  # função preparada para receber múltiplas UFs
        self.__print_df_shape(df)

        # Conversão dos tipos e tratamentos iniciais:

        logger.info("Criando o campo data_hora.") if self.verbose else None
        df["data_hora"] = pd.to_datetime(df["data_inversa"] + " " + df["horario"])
        df.drop(columns=["data_inversa", "horario"], inplace=True)

        logger.info("Padronizando os campos do tipo string.") if self.verbose else None
        df = clean_string(df, STR_COLS_TO_UPPER, "upper")
        df = clean_string(df, STR_COLS_TO_LOWER)

        # Tratamento da latitude e longitude:

        logger.info(
            "Convertendo os tipos dos campos latitude e longitude."
        ) if self.verbose else None
        df["latitude"] = (df["latitude"].str.replace(",", ".")).astype(float)
        df["longitude"] = (df["longitude"].str.replace(",", ".")).astype(float)

        logger.info(
            f"Eliminando registros com lat/lon com menos de {COORDS_MIN_DECIMAL_PLACES} casas decimais."
        ) if self.verbose else None
        df = self.__keep_min_decimal_places(df)
        self.__print_df_shape(df)

        logger.info(
            f"Mantendo somente registros de acidentes ocorridos geograficamente no {self.uf}."
        ) if self.verbose else None
        df = self.__keep_geo_correct_rows(df)
        self.__print_df_shape(df)

        logger.info(
            f"Eliminando as coordenadas outliers por delegacia."
        ) if self.verbose else None
        df = self.__remove_outlier_coords(df)
        self.__print_df_shape(df)

        # Corrige o padrão dos nomes das UOPs:
        # TODO

        # Armazena a cache caso o modo de leitura da cache não esteja ativo:
        cache_path = PATH_DATA_PRF_CACHE / f"{self.name}.pkl"
        if not self.read_cache:
            logger.info(f"Armazenado {cache_path}.")
            df.to_pickle(cache_path)

        self.df_accidents = df.copy()
        logger.info(f"Fim do pré processamento.") if self.verbose else None
        self.__print_df_shape(df)

    #######################################################################

    def __print_df_shape(self, df):
        logger.info(f"df.shape: {df.shape}")

    def __get_polygon(self):
        """Método para carregamento do json com as coordenadas e construção do polígono da região de interesse.

        Returns
        -------
        _type_
            Polígono da região de interesse.
        """
        with open(PATH_DATA_IBGE_BORDERS / f"{self.uf}.json") as file:
            borders = json.load(file)["borders"][0]

        lst_lon = [x["lng"] for x in borders]
        lst_lat = [x["lat"] for x in borders]
        polygon = Polygon(zip(lst_lon, lst_lat))

        return polygon

    def __within_polygon(self, lng: float, lat: float, polygon: Polygon) -> bool:
        """Método para identificar se um ponto está dentro de um polígono.

        Parameters
        ----------
        lng : float
            Longitude do ponto.
        lat : float
            Latitude do ponto.
        polygon : Polygon
            Polígono da região de interesse.

        Returns
        -------
        bool
            Verdadeiro se o ponto está dentro do polígono. Falso, caso contrário.
        """
        point = Point(float(lng), float(lat))
        isin_polygon = point.within(polygon)

        return isin_polygon

    def __keep_min_decimal_places(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para garantir na base somente registros com coordenadas atendendo um número mínimo de casas decimais.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame completo.

        Returns
        -------
        pd.DataFrame
            Data frame com os registros removidos.
        """
        mask_lat = get_decimal_places(df["latitude"]) >= COORDS_MIN_DECIMAL_PLACES
        mask_lon = get_decimal_places(df["longitude"]) >= COORDS_MIN_DECIMAL_PLACES
        df_out = df[mask_lat & mask_lon]

        return df_out

    def __keep_geo_correct_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para garantir registros ocorridos na região geográfica de interesse.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame completo.

        Returns
        -------
        pd.DataFrame
            Data frame com os registros removidos.
        """
        polygon = self.__get_polygon()
        isin_polygon = df.apply(
            lambda x: self.__within_polygon(x.longitude, x.latitude, polygon), axis=1
        )

        df_out = df[isin_polygon].copy()

        return df_out

    def __remove_outlier_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para remover registros de acidentes considerados outliers aos demais pontos alocados na mesma delegacia.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame completo.

        Returns
        -------
        pd.DataFrame
            Data frame com outliers removidos.
        """
        lat_abs_zscore = (
            df.groupby(["delegacia"])["latitude"]
            .transform(lambda x: stats.zscore(x, ddof=1))
            .abs()
        )
        lon_abs_zscore = (
            df.groupby(["delegacia"])["longitude"]
            .transform(lambda x: stats.zscore(x, ddof=1))
            .abs()
        )

        mask = (lat_abs_zscore <= 3) & (lon_abs_zscore <= 3)
        df_out = df[mask]

        return df_out
