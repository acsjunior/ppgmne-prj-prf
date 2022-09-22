import numpy as np
import pandas as pd
import pyomo.environ as pyo
from loguru import logger

from ppgmne_prj_prf.utils import get_distance_matrix


def get_semi_abstract_pmedian_model(
    a: int,
    u: int,
    s: int,
    dist_max: np.ndarray,
    dist_matrix: np.ndarray,
    accidents_hist: np.ndarray,
    verbose: bool = True,
) -> pyo.AbstractModel:
    """Função para gerar um modelo de p-medianas parcialmente abstrato.

    Parameters
    ----------
    a : int
        Número de pontos com acidentes e sem UOP atual.
    u : int
        Número de pontos com acidentes e com UOP atual.
    s : int
        Número de pontos sem acidentes e com UOP atual.
    dist_max : np.ndarray
        Vetor com as distâncias máximas.
    dist_matrix : np.ndarray
        Matriz de distâncias m x n.
    accidents_hist : np.ndarray
        Vetor de histórico de acidentes.
    verbose : bool, optional
       Ativa o modo logger , by default False

    Returns
    -------
    pyo.AbstractModel
        Modelo parcialmente abstrato.
    """

    dmax = dist_max
    d = dist_matrix
    h = accidents_hist

    logger.info("Instanciando o objeto.") if verbose else None
    model = pyo.AbstractModel()

    logger.info("Declarando os índices.") if verbose else None
    model.I = pyo.RangeSet(u)
    model.J = pyo.RangeSet(s)
    model.U = pyo.RangeSet(a, u)

    logger.info("Declarando os parâmetros.") if verbose else None
    model.p = pyo.Param()
    model.q = pyo.Param()
    model.d = pyo.Param(
        model.I, model.J, initialize=lambda model, i, j: d[i - 1][j - 1], mutable=True
    )
    model.h = pyo.Param(model.I, initialize=lambda model, i: h[i - 1])
    model.dmax = pyo.Param(model.I, initialize=lambda model, i: dmax[i - 1])

    logger.info("Declarando as variáveis de decisão.") if verbose else None
    model.y = pyo.Var(model.J, within=pyo.Binary)
    model.x = pyo.Var(model.I, model.J, within=pyo.Binary)

    logger.info("Declarando a função objetivo.") if verbose else None

    def f_obj(model):
        return sum(
            model.h[i] * model.d[i, j] * model.x[i, j] for i in model.I for j in model.J
        )

    model.z = pyo.Objective(rule=f_obj, sense=pyo.minimize)

    logger.info("Declarando as restrições.") if verbose else None

    def f_restr1(model, i):
        return sum(model.x[i, j] for j in model.J) == 1

    def f_restr2(model):
        return sum(model.y[j] for j in model.J) == model.p()

    def f_restr3(model):
        return sum(model.y[u] for u in model.U) >= model.q()

    def f_restr4(model, i, j):
        return model.x[i, j] <= model.y[j]

    def f_restr5(model, i, j):
        return (model.d[i, j] * model.x[i, j]) <= model.dmax[i]

    model.restr_1 = pyo.Constraint(model.I, rule=f_restr1)
    model.restr_2 = pyo.Constraint(rule=f_restr2)
    model.restr_3 = pyo.Constraint(rule=f_restr3)
    model.restr_4 = pyo.Constraint(model.I, model.J, rule=f_restr4)
    model.restr_5 = pyo.Constraint(model.I, model.J, rule=f_restr5)

    logger.info(
        "Modelo parcialmente abstrato construído com sucesso."
    ) if verbose else None

    return model


def get_dict_params(df: pd.DataFrame, p: int, q: int) -> np.ndarray:
    """Função para concentrar os parâmetros do modelo em um dicionário.

    Parameters
    ----------
    df : pd.DataFrame
        Base de quadrantes.
    p : int
        Número de medianas.
    p : int
        Número UOPs atuais a manter na solução.

    Returns
    -------
    np.ndarray
        Parâmetros do modelo.
    """
    # Matriz de distâncias:
    lat_rows = df[~df["is_only_uop"]]["latitude"]
    lon_rows = df[~df["is_only_uop"]]["longitude"]
    lat_cols = df["latitude"]
    lon_cols = df["longitude"]
    d = get_distance_matrix(lat_rows, lon_rows, lat_cols, lon_cols)

    # Demais parâmetros:
    dmax = np.array(df[~df["is_only_uop"]]["dist_max"])
    h = np.array(df[~df["is_only_uop"]]["n_accidents"])
    a = df[~df["is_uop"]].shape[0]
    u = a + df[(df["is_uop"]) & (~df["is_only_uop"])].shape[0]
    s = u + df[df["is_only_uop"]].shape[0]

    out = {}
    out["dist_matrix"] = d
    out["dist_max"] = dmax
    out["accidents_hist"] = h
    out["a"] = a
    out["u"] = u
    out["s"] = s
    out["p"] = p
    out["q"] = q

    return out


def format_params(params: dict) -> pyo.DataPortal:
    """Função para formatar um dicionário de parâmetros para criação de instâncias do modelo.

    Parameters
    ----------
    params : dict
        Dicionário de parâmetros.

    Returns
    -------
    pyo.DataPortal
        Parâmetros formatados.
    """

    def format_param(param):
        if isinstance(param, list):
            if isinstance(param[0], list):
                param = [tuple(row) for row in param]
        return {None: param}

    out = pyo.DataPortal()
    for key in params:
        out[key] = format_param(params[key])

    return out


def get_solution_data(instance: pyo.ConcreteModel, df: pd.DataFrame) -> pd.DataFrame:
    """Função para extrair os resultados do modelo.

    Parameters
    ----------
    instance : pyo.ConcreteModel
        Modelo.
    df : pd.DataFrame
        Base de quadrantes.

    Returns
    -------
    pd.DataFrame
        Base de quadrantes somente com os pontos de demanda e com os resultados do modelo.
    """

    # Adiciona a flag identificadora se o ponto é uma mediana:
    df["is_median"] = [int(instance.y[j]()) == 1 for j in list(instance.y.keys())]

    # Separa em dados de medianas e dados de demanda:
    df_demand = df[~df["is_only_uop"]].copy()

    df_median = df[df["is_median"]][["name", "latitude", "longitude"]].copy()
    df_median.rename(
        columns={
            "name": "median_name",
            "latitude": "median_lat",
            "longitude": "median_lon",
        },
        inplace=True,
    )

    # Adiciona os resultados no dataset de demanda:
    aloc_tuple = [x for x in list(instance.x.keys()) if instance.x[x]() == 1]
    aloc_tuple.sort(key=lambda x: x[0])
    df_demand["median_name"] = [df_demand["name"][tupla[1] - 1] for tupla in aloc_tuple]
    df_demand["distance_q_to_m"] = [instance.d[x[0], x[1]]() for x in aloc_tuple]

    # Adiciona as coordenadas das medianas:
    df_demand = df_demand.merge(df_median, on=["median_name"])
    df_demand.sort_values(by=["municipality", "name"], inplace=True)

    return df_demand
