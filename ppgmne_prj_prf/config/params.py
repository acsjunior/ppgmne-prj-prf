# Parâmetros dos acidentes:
UF = "PR"

STR_COLS_TO_LOWER = [
    "dia_semana",
    "causa_acidente",
    "tipo_acidente",
    "classificacao_acidente",
    "fase_dia",
    "sentido_via",
    "tipo_pista",
    "tracado_via",
    "uso_solo",
]
STR_COLS_TO_UPPER = ["municipio", "regional", "delegacia", "uop"]

COORDS_MIN_DECIMAL_PLACES = 3
COORDS_PRECISION = 2

CLUSTERING_FEATS = ["point_n_accidents"]
N_CLUSTERS = 8
CLUSTER_DMAX = {1: 180, 2: 180, 3: 120, 4: 120, 5: 120, 6: 60, 7: 60, 8: 60}

# Parâmetros do IBGE
IBGE_YEAR = 2019
