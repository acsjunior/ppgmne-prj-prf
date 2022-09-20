import json

import kml2geojson
import numpy as np
import pandas as pd
from loguru import logger
from unidecode import unidecode

from ppgmne_prj_prf.config.params import UF
from ppgmne_prj_prf.config.paths import PATH_DATA_PRF, PATH_DATA_PRF_CACHE_DATABASE
from ppgmne_prj_prf.utils import concatenate_dict_of_dicts, trace_df


class Stations:
    def __init__(self, verbose: bool = True, read_cache: bool = False):
        self.name_raw = "stations_raw"
        self.name = "stations"
        self.uf = UF
        self.df_stations = pd.DataFrame()
        self.verbose = verbose
        self.read_cache = read_cache

    def extract(self):
        """Método para extração dos dados das estações policiais"""

        logger.info(
            "Início da leitura dos dados das estações policiais."
        ) if self.verbose else None

        cache_path = PATH_DATA_PRF_CACHE_DATABASE / f"{self.name_raw}.pkl"

        if self.read_cache:
            logger.info("Modo de leitura da cache ativo.")
            self.__geo_stations = pd.read_pickle(cache_path)
            logger.info("Fim da extração dos dados.")
            return

        logger.info(
            "Convertendo os dados das estações policiais para GeoJson."
        ) if self.verbose else None
        self.__geo_stations = kml2geojson.main.convert(PATH_DATA_PRF / f"{self.uf}.kml")

        logger.info(f"Fim da extração dos dados.")

    def transform(self):
        """Método para o pré-processamento dos dados das estações policiais"""

        logger.info("Início do pré processamento.") if self.verbose else None

        cache_path = PATH_DATA_PRF_CACHE_DATABASE / f"{self.name}.pkl"

        if self.read_cache:
            logger.info("Modo de leitura da cache ativo.")
            self.df_stations = pd.read_pickle(cache_path)
            logger.info("Fim do pré-processamento.")
            return

        df_out = self.__structure_stations().pipe(trace_df)

        logger.info("Incluindo os códigos das UOPs.") if self.verbose else None
        with open(PATH_DATA_PRF / "transformations.json") as file:
            stations_to_replace = concatenate_dict_of_dicts(
                json.load(file)["stations_replace"]
            )
        df_out["uop"] = df_out["name"].map(stations_to_replace)

        # Armazena a cache caso o modo de leitura da cache não esteja ativo:
        if not self.read_cache:
            logger.info(f"Armazenado {cache_path}.")
            df_out.to_pickle(cache_path)

        self.df_stations = df_out.pipe(trace_df)
        logger.info("Fim do pré-processamento.") if self.verbose else None

    ############################################################################################################

    def __structure_stations(self) -> pd.DataFrame:

        logger.info(
            "Estruturando os dados das estações policiais."
        ) if self.verbose else None

        dict_chars = {
            "type": [],
            "name": [],
            "station_father": [],
            "station_code": [],
            "address": [],
            "municipality": [],
            "state": [],
            "phone": [],
            "email_del": [],
            "email_uop": [],
            "latitude": [],
            "longitude": [],
        }

        lst_full_description = []

        for d in self.__geo_stations[0]["features"]:
            if d["type"] == "Feature":

                # Extrai as informações parcialmente tratadas:
                full_description = d["properties"]["description"].split("<br>")
                longitude = float(
                    str(d["geometry"]["coordinates"][0]).replace(",", ".")
                )
                latitude = float(str(d["geometry"]["coordinates"][1]).replace(",", "."))

                # Insere as informações iniciais no dicionário:
                dict_chars["longitude"].append(longitude)
                dict_chars["latitude"].append(latitude)

                # Insere a descrição completa em uma lista temporária:
                lst_full_description.append(full_description)

                # Extrai as informações detalhadas da descrição:
                for x in lst_full_description:
                    name = unidecode(x[0]).strip().upper()

                    if "SUPERINTENDENCIA" in name:
                        type = "SPRF"
                    elif "DELEGACIA" in name:
                        type = "DEL"
                    elif "UOP" in name:
                        type = "UOP"
                    else:
                        type = "other"

                    address = x[1]

                    municipality = unidecode(x[2].split("/")[0]).strip().upper()
                    state = unidecode(x[2].split("/")[1]).strip().upper()

                    phone = x[4].strip().lower().replace("telefone:", "").strip()

                    email_del = x[5].strip().lower().replace("email:", "").strip()

                    if len(x) == 7:
                        email_uop = np.nan
                    else:
                        email_uop = x[6].strip().lower()

                # Extrai o código do posto a partir do email_del:
                station_father = np.nan
                if type == "SPRF":
                    station_father = type
                elif not pd.isnull(email_del):
                    station_father = email_del.upper().split(".")[0]

                # Extrai o código do posto a partir do email_uop:
                station_code = np.nan
                if not pd.isnull(email_uop):
                    station_code = email_uop.upper().split(".")[0]

                # Insere as informações finais no dicionário:
                dict_chars["type"].append(type)
                dict_chars["name"].append(name)
                dict_chars["station_father"].append(station_father)
                dict_chars["station_code"].append(station_code)
                dict_chars["address"].append(address)
                dict_chars["municipality"].append(municipality)
                dict_chars["state"].append(state)
                dict_chars["phone"].append(phone)
                dict_chars["email_del"].append(email_del)
                dict_chars["email_uop"].append(email_uop)

        df_out = pd.DataFrame(dict_chars)

        return df_out
