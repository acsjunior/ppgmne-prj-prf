import json

import holidays
import numpy as np
import pandas as pd
import scipy.stats as stats
from loguru import logger
from pandera import Column, DataFrameSchema
from pandera.errors import SchemaError
from shapely.geometry import Point, Polygon
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from ppgmne_prj_prf.config.params import (
    CLUSTERING_FEATS,
    COORDS_MIN_DECIMAL_PLACES,
    COORDS_PRECISION,
    N_CLUSTERS,
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
        self.df_accidents = pd.DataFrame()
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
        """Método para extração dos do histórico de acidentes"""

        logger.info(
            "Início da leitura dos registros de acidentes."
        ) if self.verbose else None

        cache_path = PATH_DATA_PRF_CACHE / f"{self.name_raw}.pkl"

        if self.read_cache:
            logger.info("Modo de leitura da cache ativo.")
            self.df_accidents = pd.read_pickle(cache_path)
            logger.info("Fim da extração dos dados.")
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
        """Método para pré-processamento do histórico de acidentes"""

        logger.info("Início do pré processamento.") if self.verbose else None

        cache_path = PATH_DATA_PRF_CACHE / f"{self.name}.pkl"
        if self.read_cache:
            logger.info("Modo de leitura da cache ativo.")
            self.df_accidents = pd.read_pickle(cache_path)
            logger.info("Fim do pré-processamento.")
            return

        df = self.df_accidents
        self.__print_df_shape(df)

        logger.info("Removendo os registros incompletos.") if self.verbose else None
        df.dropna(inplace=True)
        self.__print_df_shape(df)

        # Mantém na base somente registros nas delegacias da UF desejada:
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

        logger.info(
            "Criando as flags de feriado e final de semana."
        ) if self.verbose else None
        df = self.__classify_holidays(df)
        df["is_weekend"] = df["data_hora"].dt.weekday >= 5

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

        logger.info(f"Aplicação das correções manuais.") if self.verbose else None
        df = self.__manual_transformations(df)
        self.__print_df_shape(df)

        logger.info(
            f"Eliminando as coordenadas outliers por delegacia."
        ) if self.verbose else None
        df = self.__remove_outlier_coords(df)
        self.__print_df_shape(df)

        logger.info(
            f"Criando as coordendas dos pontos (arredondamento)."
        ) if self.verbose else None
        df["point_lat"] = df["latitude"].round(COORDS_PRECISION)
        df["point_lon"] = df["longitude"].round(COORDS_PRECISION)

        logger.info(f"Criando a identificação dos pontos.") if self.verbose else None
        df = self.__identify_point(df)
        self.__print_df_shape(df)

        logger.info(f"Calculando as estatísticas dos pontos.") if self.verbose else None
        df = self.__get_point_stats(df)
        self.__print_df_shape(df)

        logger.info(f"Identifica o cluster de cada ponto.") if self.verbose else None
        df = self.__get_point_clusters(df)
        self.__print_df_shape(df)

        # Armazena a cache caso o modo de leitura da cache não esteja ativo:
        cache_path = PATH_DATA_PRF_CACHE / f"{self.name}.pkl"
        if not self.read_cache:
            logger.info(f"Armazenado {cache_path}.")
            df.to_pickle(cache_path)

        self.df_accidents = df.copy()
        logger.info(f"Fim do pré-processamento.") if self.verbose else None
        self.__print_df_shape(df)

    #######################################################################

    def __print_df_shape(self, df: pd.DataFrame):
        """Método para impressão das dimensões de um data frame

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        """
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

    def __manual_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para aplicar as correções necessárias identificadas após análise.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame dos acidentes.

        Returns
        -------
        pd.DataFrame
            Data frame com as correções aplicadas.
        """
        # Lê o json com as correções manuais:
        with open(PATH_DATA_PRF / "transformations.json") as file:
            transformations = json.load(file)

            accidents_to_delete_by_uop = transformations["accidents_deletion"]["uop"]
            uops_to_replace = transformations["accidents_replace"]["uop"]
            dels_to_replace = transformations["accidents_replace"]["del"]

        # Deleta os registros a serem desconsiderados:
        df_out = df[~df["uop"].isin(accidents_to_delete_by_uop)].copy()

        # Corrige os registros:
        right_dels = df_out["uop"].map(dels_to_replace)
        right_uops = df_out["uop"].map(uops_to_replace)
        df_out["delegacia"] = right_dels.combine_first(df_out["delegacia"])
        df_out["uop"] = right_uops.combine_first(df_out["uop"])

        return df_out

    def __classify_holidays(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para classificar uma data como feriado (True) ou não (False).

        Parameters
        ----------
        df : pd.DataFrame
            Data frame completo.

        Returns
        -------
        pd.DataFrame
            Data frame com a flag "is_holiday".
        """
        br_holidays = holidays.country_holidays("BR", subdiv=self.uf)
        df["is_holiday"] = ~(df["data_hora"].apply(br_holidays.get)).isna()

        return df

    def __identify_point(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para criar identificação única e padronizar o nome do município dos pontos.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame completo.

        Returns
        -------
        pd.DataFrame
            Data frame com os campos "point_mun" e "point_name".
        """

        # Identifica o município do ponto por ordem de frequência de acidentes:
        df_mun = (
            df.groupby(["point_lat", "point_lon", "municipio"])["data_hora"]
            .count()
            .reset_index(name="n_accidents")
        )
        df_mun["seq"] = df_mun.groupby(["point_lat", "point_lon"])["n_accidents"].rank(
            "first", ascending=False
        )
        df_mun = df_mun[df_mun["seq"] == 1].copy()
        df_mun.rename(columns={"municipio": "point_mun"}, inplace=True)

        # Cria um identificador único para o ponto:
        zfill_param = len(
            str(
                df_mun.groupby(["point_mun"])["point_lat"]
                .count()
                .reset_index(name="n")["n"]
                .max()
            )
        )
        df_mun.sort_values(by=["point_mun", "point_lat", "point_lon"], inplace=True)
        df_mun["x"] = 1
        df_mun["suf"] = (
            (df_mun.groupby("point_mun")["x"].rank("first"))
            .astype(int)
            .astype(str)
            .str.zfill(zfill_param)
        )
        df_mun["point_name"] = df_mun["point_mun"] + " " + df_mun["suf"]

        # Remove os campos desnecessários:
        df_mun.drop(columns=["n_accidents", "seq", "x", "suf"], inplace=True)

        # Inclui os campos no df final:
        df_out = df.merge(df_mun, on=["point_lat", "point_lon"])

        return df_out

    def __get_point_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para calcular as estatísticas por ponto.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame completo.

        Returns
        -------
        pd.DataFrame
            Data frame com as estatísticas calculadas.
        """
        # Calcula as estatísticas:
        df_stats = (
            df.groupby(["point_name"])
            .agg(
                point_acc=("data_hora", "count"),
                point_acc_holiday=("is_holiday", sum),
                point_acc_weekend=("is_weekend", sum),
                point_inj=("feridos_graves", sum),
                point_dead=("mortos", sum),
            )
            .reset_index()
        )

        # Inclui os dados no df final:
        df_out = df.merge(df_stats, on="point_name")

        return df_out

    def __get_point_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para clusterização dos pontos de acidentes.

        Aplica o método hierárquico de Ward.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame completo.

        Returns
        -------
        pd.DataFrame
            Data frame com o cluster de cada ponto.
        """

        df_point = (
            df[["point_name"] + CLUSTERING_FEATS]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        df_cluster = df_point[CLUSTERING_FEATS].copy()
        df_cluster.iloc[:, :] = StandardScaler().fit_transform(df_cluster)

        hc = AgglomerativeClustering(
            n_clusters=N_CLUSTERS, affinity="euclidean", linkage="ward"
        )

        df_point["cluster"] = (hc.fit_predict(df_cluster)).astype(str)

        # Calcula as estatísticas por cluster:
        df_stats = None
        for cluster in df_point["cluster"].value_counts().index:
            df_stats_i = pd.DataFrame(
                df_point[df_point["cluster"] == cluster]["point_acc"].describe()
            ).T
            df_stats_i["cluster"] = cluster
            if df_stats is None:
                df_stats = df_stats_i.copy()
            else:
                df_stats = pd.concat([df_stats, df_stats_i])
        df_stats = df_stats.sort_values(by="mean").reset_index(drop=True)

        # Armazena as estatísticas:
        # TODO

        # Renomeia os clusters:
        clusters = np.arange(1, N_CLUSTERS + 1, 1)
        df_stats["point_cluster"] = clusters
        df_stats["point_cluster"] = pd.Categorical(
            df_stats["point_cluster"], categories=clusters, ordered=True
        )

        # Inclui os clusters renomeados na base de pontos:
        df_point = df_point.merge(df_stats[["cluster", "point_cluster"]], on="cluster")

        # Armazena a base de clusters:
        # TODO

        # Inclui os clusters na base final:
        df_out = df.merge(df_point[["point_name", "point_cluster"]], on="point_name")

        return df_out
