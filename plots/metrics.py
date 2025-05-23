import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import kendalltau

def init_metrics() -> dict:
    return {"mse": [], "kendall": [], "cosine": [], "r2": []}

def calculate_mse(actual: np.ndarray, estimate: np.ndarray) -> float:
    return np.square(np.subtract(actual, estimate)).mean()

def calculate_kendall(actual: np.ndarray, estimate: np.ndarray) -> float:
    tau, _ = kendalltau(actual, estimate)
    return tau

def calculate_cosine(actual: np.ndarray, estimate: np.ndarray) -> float:
    # Reshape to 2D for cosine similarity
    actual_reshaped = actual.reshape(1, -1)
    estimate_reshaped = estimate.reshape(1, -1)
    return cosine_similarity(actual_reshaped, estimate_reshaped)[0][0]

def calculate_r2(actual: np.ndarray, estimate: np.ndarray) -> float:
    return r2_score(actual, estimate)

def calculate_metrics(actual: np.ndarray, estimate: np.ndarray) -> dict:

    estimate= np.round(estimate, 10)
    actual = np.round(actual, 10)

    return {
        "mse": calculate_mse(actual, estimate),
        "kendall": calculate_kendall(actual, estimate),
        "cosine": calculate_cosine(actual, estimate),
        "r2": calculate_r2(actual, estimate)
    }
    
def update_metrics(metrics: dict, new_metrics: dict) -> None:
    
    for key in metrics.keys():
        metrics[key].append(new_metrics[key])

    return metrics

def get_sorted_indices(estimate: np.ndarray) -> np.ndarray:
    
    return np.argsort(estimate)

def compute_sra_all_k(actual:np.ndarray, estimate:np.ndarray) -> np.ndarray:
    assert actual.shape == estimate.shape, "Arrays must be of the same length"
    n = actual.shape[0]

    estimate= np.round(estimate, 10)
    actual = np.round(actual, 10)

    # Sort indices by descending absolute value
    estimate_order = np.argsort(-np.abs(estimate))
    actual_order = np.argsort(-np.abs(actual))

    estimate_ranks = np.empty(n, dtype=int)
    actual_ranks = np.empty(n, dtype=int)
    for rank, idx in enumerate(estimate_order):
        estimate_ranks[idx] = rank
    for rank, idx in enumerate(actual_order):
        actual_ranks[idx] = rank

    estimate_signs = np.sign(estimate)
    actual_signs = np.sign(actual)

    # SRA@k for all k
    sra_k = np.zeros(n)
    for k in range(1, n+1):
        top_k_estimate = set(estimate_order[:k])
        top_k_actual = set(actual_order[:k])
        common = top_k_estimate & top_k_actual      

        if not common:
            sra_k[k-1] = 0
            continue

        count = 0
        for j in common:

            if estimate_ranks[j] == actual_ranks[j] and estimate_signs[j] == actual_signs[j]:
            # if estimate_ranks[j] == actual_ranks[j]:
                count += 1
        sra_k[k-1] = count / k

    return sra_k
