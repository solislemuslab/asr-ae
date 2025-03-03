import numpy as np
import pandas as pd
from typing import Tuple

def read_params_from_PAML(file_path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Output: Tuple(exchangeabilities, stationary frequencies)
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # First 20 lines contain the exchange matrix
    exchange_lines = lines[:20]
    # Convert to a list of floats
    exchanges = []
    for line in exchange_lines:
        row = list(map(float, line.split()))
        exchanges.append(row)
    q = len(exchanges)
    # Convert to a symmetric matrix
    S = np.zeros((q, q), dtype=float)
    for i, row in enumerate(exchanges):
        S[i, :len(row)] = row
    S = S + S.T 
    
    # 22nd line contains the frequencies
    freqs = list(map(float, lines[21].split()))
    freqs = np.array(freqs)
    return S, freqs

def read_rate_matrix_from_csv(rate_matrix_path: str) -> pd.DataFrame:
    """
    Output: rate matrix as a pandas DataFrame
    """
    res = pd.read_csv(
        rate_matrix_path,
        sep='\s+',
        index_col=0,
        keep_default_na=False,
        na_values=["_"],
    ).astype(float)
    # check that rows sum to 0
    if not np.allclose(res.sum(axis=1), 0, atol=1e-6):
        raise Exception(
            f"Rate matrix at {rate_matrix_path} should have rows summing to 0."
        )
    return res

def read_probability_distribution_from_csv(
    probability_distribution_path: str,
) -> pd.DataFrame:
    """
    Output: probability distribution as a pandas DataFrame
    """
    res = pd.read_csv(
        probability_distribution_path,
        sep='\s+',
        index_col=0,
        keep_default_na=False,
        na_values=["_"],
    ).astype(float)
    if res.shape[1] != 1:
        raise Exception(
            f"Probability distribution at {probability_distribution_path} "
            "should be one-dimensional."
        )
    if abs(res.sum().sum() - 1.0) > 1e-6:
        raise Exception(
            f"Probability distribution at {probability_distribution_path} "
            "should add to 1.0, with a tolerance of 1e-6."
        )
    return res

def compute_scale(rate_matrix, eq_probs):
    diago = np.diag(rate_matrix)
    tmp = np.multiply(diago, eq_probs)
    scale = -np.sum(tmp)
    return scale