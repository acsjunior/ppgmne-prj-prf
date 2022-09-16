import json

import pandas as pd
from loguru import logger
from pandera import Column, DataFrameSchema
from pandera.errors import SchemaError

from ppgmne_prj_prf.config.params import UF
from ppgmne_prj_prf.config.paths import PATH_DATA_PRF, PATH_DATA_PRF_CACHE
from ppgmne_prj_prf.database.utils import csv_zip_to_df


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
        logger.info(f"Fim!")

    def transform(self):
        None
