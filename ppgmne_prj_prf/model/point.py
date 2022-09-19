import pickle

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

from ppgmne_prj_prf.config.params import (
    CLUSTER_DMAX,
    CLUSTERING_FEATS,
    COORDS_PRECISION,
    N_CLUSTERS,
)
from ppgmne_prj_prf.config.paths import PATH_DATA_PRF_CACHE_MODEL


class Point:
    def __init__(
        self,
        df_accidents: pd.DataFrame,
        df_stations: pd.DataFrame,
        verbose: bool = True,
        read_cache: bool = False,
        cluster_cache: bool = False,
    ):
        self.name = "points"
        self.df_accidents = df_accidents
        self.df_stations = df_stations
        self.df_point = pd.DataFrame()
        self.verbose = verbose
        self.read_cache = read_cache
        self.cluster_cache = cluster_cache

    def transform(self):
        """Método para pré-processamento dos pontos de acidentes"""

        logger.info("Início do pré processamento.") if self.verbose else None
        df = self.df_accidents.copy()

        cache_path = PATH_DATA_PRF_CACHE_MODEL / f"{self.name}.pkl"
        if self.read_cache:
            logger.info("Modo de leitura da cache ativo.")
            self.df_point = pd.read_pickle(cache_path)
            logger.info("Fim do pré-processamento.")
            return

        logger.info(
            f"Criando as coordendas dos pontos (arredondamento)."
        ) if self.verbose else None
        df["point_latitude"] = df["latitude"].round(COORDS_PRECISION)
        df["point_longitude"] = df["longitude"].round(COORDS_PRECISION)

        logger.info(f"Criando a identificação dos pontos.") if self.verbose else None
        df = self.__identify_point(df)
        self.__print_df_shape(df)

        logger.info(f"Calculando as estatísticas dos pontos.") if self.verbose else None
        df = self.__get_point_stats(df)
        self.__print_df_shape(df)

        logger.info(f"Identificando o cluster de cada ponto.") if self.verbose else None
        df = self.__get_point_clusters(df)
        self.__print_df_shape(df)

        logger.info(f"Agregando os dados por ponto.") if self.verbose else None
        df = self.__aggregate_points(df)
        self.__print_df_shape(df)

        logger.info(f"Incluindo o DMAX na base.") if self.verbose else None
        df["dist_max"] = df["cluster"].map(CLUSTER_DMAX)

        # Armazena a cache caso o modo de leitura da cache não esteja ativo:
        if not self.read_cache:
            logger.info(f"Armazenado {cache_path}.")
            df.to_pickle(cache_path)

        self.df_point = df.copy()
        logger.info(f"Fim do pré-processamento.") if self.verbose else None
        self.__print_df_shape(df)

    #################################################################
    def __print_df_shape(self, df: pd.DataFrame):
        """Método para impressão das dimensões de um data frame

        Parameters
        ----------
        df : pd.DataFrame
            _description_
        """
        logger.info(f"df.shape: {df.shape}")

    def __identify_point(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para criar identificação única e padronizar o nome do município dos pontos.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame completo.

        Returns
        -------
        pd.DataFrame
            Data frame com os campos "point_municipality" e "point_name".
        """

        # Identifica o município do ponto por ordem de frequência de acidentes:
        df_mun = (
            df.groupby(["point_latitude", "point_longitude", "municipio"])["data_hora"]
            .count()
            .reset_index(name="n_accidents")
        )
        df_mun["seq"] = df_mun.groupby(["point_latitude", "point_longitude"])[
            "n_accidents"
        ].rank("first", ascending=False)
        df_mun = df_mun[df_mun["seq"] == 1].copy()
        df_mun.rename(columns={"municipio": "point_municipality"}, inplace=True)

        # Cria um identificador único para o ponto:
        zfill_param = len(
            str(
                df_mun.groupby(["point_municipality"])["point_latitude"]
                .count()
                .reset_index(name="n")["n"]
                .max()
            )
        )
        df_mun.sort_values(
            by=["point_municipality", "point_latitude", "point_longitude"], inplace=True
        )
        df_mun["x"] = 1
        df_mun["suf"] = (
            (df_mun.groupby("point_municipality")["x"].rank("first"))
            .astype(int)
            .astype(str)
            .str.zfill(zfill_param)
        )
        df_mun["point_name"] = df_mun["point_municipality"] + " " + df_mun["suf"]

        # Remove os campos desnecessários:
        df_mun.drop(columns=["n_accidents", "seq", "x", "suf"], inplace=True)

        # Inclui os campos no df final:
        df_out = df.merge(df_mun, on=["point_latitude", "point_longitude"])

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
                point_n_accidents=("data_hora", "count"),
                point_n_acc_holiday=("is_holiday", sum),
                point_n_acc_weekend=("is_weekend", sum),
                point_n_injuried=("feridos_graves", sum),
                point_n_dead=("mortos", sum),
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

        # Se a leitura de cache estiver ativa, carrega a base de clusters da cache:
        clusters_path = PATH_DATA_PRF_CACHE_MODEL / f"hc_clusters.pkl"
        if self.cluster_cache:
            logger.info("Lendo a cache da base de clusters.")
            df_point = pd.read_pickle(clusters_path)

            # Inclui os clusters na base final:
            df_out = df.merge(
                df_point[["point_name", "point_cluster"]], on="point_name"
            )
            return df_out

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

        # Armazena o pickle do modelo caso o modo de leitura da cache não esteja ativo:
        model_path = PATH_DATA_PRF_CACHE_MODEL / "hc_model.pkl"
        if not self.cluster_cache:
            logger.info(f"Armazenado o modelo de clustering em {model_path}.")
            pickle.dump(hc, open(model_path, "wb"))

        df_point["cluster"] = (hc.fit_predict(df_cluster)).astype(str)

        # Calcula as estatísticas por cluster:
        df_stats = None
        for cluster in df_point["cluster"].value_counts().index:
            df_stats_i = pd.DataFrame(
                df_point[df_point["cluster"] == cluster]["point_n_accidents"].describe()
            ).T
            df_stats_i["cluster"] = cluster
            if df_stats is None:
                df_stats = df_stats_i.copy()
            else:
                df_stats = pd.concat([df_stats, df_stats_i])
        df_stats = df_stats.sort_values(by="mean").reset_index(drop=True)

        # Armazena as estatísticas caso o modo de leitura da cache não esteja ativo:
        stats_path = PATH_DATA_PRF_CACHE_MODEL / f"hc_stats.pkl"
        if not self.cluster_cache:
            logger.info(f"Armazenado as estatísticas de clustering em {stats_path}.")
            df_stats.to_pickle(stats_path)

        # Renomeia os clusters:
        clusters = np.arange(1, N_CLUSTERS + 1, 1)
        df_stats["point_cluster"] = clusters
        df_stats["point_cluster"] = pd.Categorical(
            df_stats["point_cluster"], categories=clusters, ordered=True
        )

        # Inclui os clusters renomeados na base de pontos:
        df_point = df_point.merge(df_stats[["cluster", "point_cluster"]], on="cluster")

        # Armazena a base de clusters caso o modo de leitura da cache não esteja ativo:
        if not self.cluster_cache:
            logger.info(f"Armazenado a base de clusters em {clusters_path}.")
            df_point[["point_name", "point_cluster"]].to_pickle(clusters_path)

        # Inclui os clusters na base final:
        df_out = df.merge(df_point[["point_name", "point_cluster"]], on="point_name")

        return df_out

    def __aggregate_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Método para criar a base de pontos agregados.

        Parameters
        ----------
        df : pd.DataFrame
            Data frame na granularidade de acidentes.

        Returns
        -------
        pd.DataFrame
            Data frame agregado por ponto.
        """
        suffix = "point_"
        cols = [col for col in df.columns if col[: len(suffix)] == suffix]

        df_out = df[cols].drop_duplicates().reset_index(drop=True)
        df_out.columns = [col.replace(suffix, "") for col in cols]

        return df_out
