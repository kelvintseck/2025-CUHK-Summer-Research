import os 
import numpy as np
from numba import jit
from IPython.display import display
from scipy.optimize import minimize
from scipy.special import gamma as gammaFunc
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")


# Risk Measurements------------------------------------------------------------------------------------------------------------------------------------
def tailProb(N: int, nu: float, samples: np.ndarray) -> float:
    return np.sum(samples > nu) / N

def expExReturn(N: int, nu: float, samples: np.ndarray, h) -> float:
    excesses = h(samples[samples > nu]) - h(nu)
    return np.sum(excesses[excesses >= 0]) / N

def valueAtRisk(N: int, nu: float, samples: np.ndarray) -> float:
    sorted_samples = np.sort(samples)
    index = int(np.ceil((1 - 1/nu) * N)) - 1
    return sorted_samples[index] if 0 <= index < N else np.inf

def condVAR(N: int, nu: float, samples: np.ndarray) -> float:
    var = valueAtRisk(N, nu, samples)
    return var + nu * np.mean(samples[samples > var] - var) if np.any(samples > var) else var
#------------------------------------------------------------------------------------------------------------------------------------------------------


# Functions for updating parameters--------------------------------------------------------------------------------------------------------------------
def betaUpdate(samples: np.ndarray, gamma: float, N: int) -> float:
    valid_samples = samples[samples > gamma]
    return np.sum(np.log(valid_samples / gamma)) / N if len(valid_samples) > 0 else 0.0

def gammaUpdate(T: int, delta: float, samples: np.ndarray, N: int) -> float:
    target_index = int(N * (1 - T ** (delta - 1)))
    if target_index >= N:
        return np.min(samples)
    if target_index < 0:
        return np.max(samples)
    return -np.partition(-samples, target_index)[target_index]  # Return the largest gamma

def rateFunc(alpha: np.ndarray, beta: np.ndarray) -> float:
    def KLdiver(a, b):
        return np.where(b > 1e-10, a/b - np.log(np.clip(a/b, 1e-10, np.inf)) - 1, np.inf)
    idx_b = np.argmin(beta)
    alpha_b, beta_b = alpha[idx_b], beta[idx_b]
    mask = (beta != beta_b) & (beta > 0)
    if not np.any(mask):
        return np.inf
    ratio = (alpha_b + alpha[mask]) / (alpha_b/beta_b + alpha[mask]/beta[mask])
    vals = alpha_b * KLdiver(ratio, beta_b) + alpha[mask] * KLdiver(ratio, beta[mask])
    return -np.min(vals)
    
# Estimation of alpha_star
def alphaHatUpdate(k: int, beta: np.array) -> np.array:
    init = np.ones(k)/k
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, None) for _ in range(k)]  # alpha_i >= 0
    # result = minimize(rateFunc, init, args = (beta), bounds = bounds, constraints = constraints, method = "SLSQP")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = minimize(rateFunc, init, args=(beta), bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 50})
    return result.x

# Sampling decision
def alphaBarUpdate(t: int, m: int, alpha_hat: np.ndarray, k: int, alpha_pi: np.ndarray) -> np.ndarray:
    def norm(x: np.ndarray, t, m, alpha_hat, alpha_pi):
        return np.linalg.norm((t + m) * alpha_hat - (t * alpha_pi + m * x))
    init = np.ones(k) / k
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
    bounds = [(0, None)] * k
    # result = minimize(norm, init, args=(t, m, alpha_hat, alpha_pi), bounds=bounds, constraints=constraints, method="SLSQP")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        result = minimize(norm, init, args=(t, m, alpha_hat, alpha_pi), bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 50})
    return result.x

def distributeM(alpha_bar: np.ndarray, m: int) -> np.ndarray:
    m_i = np.round(m * alpha_bar).astype(int)
    diff = m - np.sum(m_i)
    if diff > 0:
        m_i[np.argmax(m_i)] += diff
    elif diff < 0:
        m_i[np.argmax(m_i)] -= -diff
    return m_i
#------------------------------------------------------------------------------------------------------------------------------------------------------



# Main algorithm --------------------------------------------------------------------------------------------------------------------------------------
def pfs(pred: np.ndarray, actual: int = 1) -> float:
    return np.mean(pred != actual)


def TIRO(k: int, delta: float, n0: int, m: int, T: int, dists: list) -> int:
    t = k * n0
    alpha_pi = np.zeros(k)
    alpha_bar = np.zeros(k)
    gamma = np.zeros(k)
    beta = np.zeros(k)
    allSamples = [np.array(dists[i](n0, i+1)) for i in range(k)]
    
    while t < T + 1:
        alpha_pi += alpha_bar
        gamma = np.array([gammaUpdate(T, delta, allSamples[i], len(allSamples[i])) for i in range(k)])
        beta = np.array([betaUpdate(allSamples[i], gamma[i], len(allSamples[i])) for i in range(k)])
        alpha_hat = alphaHatUpdate(k, beta)
        alpha_bar = alphaBarUpdate(t, m, alpha_hat, k, alpha_pi)
        distM = distributeM(alpha_bar, m)
        new_samples = [np.array(dists[i](distM[i], i+1)) for i in range(k)]
        allSamples = [np.concatenate((allSamples[i], new_samples[i])) for i in range(k)]
        t += m
    
    gamma = np.array([gammaUpdate(T, delta, allSamples[i], len(allSamples[i])) for i in range(k)])
    beta = np.array([betaUpdate(allSamples[i], gamma[i], len(allSamples[i])) for i in range(k)])
    
    return np.argmin(beta) + 1
#------------------------------------------------------------------------------------------------------------------------------------------------------



# Distribution functions-------------------------------------------------------------------------------------------------------------------------------
# Pareto Distribution (Type I)
def generatePareto(m, i):
    kappa = 1/(0.2 + 0.025 * i)
    tau = 1 - 1/kappa
    u = np.random.uniform(0, 1, m)
    samples = tau * (1 - u) ** (-1/kappa)
    return samples

# Student's t Distribution
expAbsVals = {i: np.mean(np.abs(np.random.standard_t(1/(0.25 + 0.05 * i), 100000))) for i in range(1, 11)}
def generateT(m, i):
    omega = 1/(0.25 + 0.05 * i)
    nu = (np.pi * omega) ** (-1/2) * gammaFunc((omega+1)/2) / gammaFunc(omega/2)
    z = np.random.standard_t(omega, m)
    samples = np.abs(z) + 3 - expAbsVals[i]
    return samples

# Fréchet Distribution
def generateFrechet(m, i):
    alpha = 1/(0.225 + 0.025 * i)
    s = 1/gammaFunc(1 - 1/alpha)
    u = np.random.uniform(0, 1, m)
    samples = s * (-np.log(u)) ** (-1/alpha)
    return samples

distPareto = [lambda m, idx=i: generatePareto(m, idx) for i in range(1, 11)]
distT = [lambda m, idx=i: generateT(m, idx) for i in range(1, 11)]
distFrechet = [lambda m, idx=i: generateFrechet(m, idx) for i in range(1, 11)]
# i = 1 is optimal in all cases.
#------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
    
        results = np.zeros((3, 10))
        for i, (distFunc, distName) in enumerate([(distPareto, "Pareto"), (distT, "Student-t"), (distFrechet, "Frechet")]):
            print(f"{distName}:")
            for idx, T in enumerate(range(1000, 10001, 1000)):
                print(f"    T = {T}:")
                initTime = time.time()
                pred = Parallel(n_jobs=-1)(delayed(TIRO)(k=10, delta=0.8, n0=100, m=100, T=T, dists=distPareto) for _ in range(10000))
                results[i][idx] = pfs(np.array(pred))
                print(f"        PFS: {results[i][idx]}")
                print(f"        Time used: {time.time() - initTime}")

            plt.plot(range(1000, 10001, 1000), results[i])
            plt.xlabel("T")
            plt.ylabel("PFS")
            plt.title(f"TRIO on {distName}")
            plt.savefig(os.path.join(curr_dir, f"TRIO_{distName}_original.png"))
        
        
        for i, distName in enumerate(["Pareto", "Student-t", "Frechet"]):
            plt.plot(range(1000, 10001, 1000), results[i], label = f"{distName}")
        plt.xlabel("T")
        plt.ylabel("PFS")
        plt.title("TRIO")
        plt.legend()
        plt.savefig(os.path.join(curr_dir, f"TRIO_original.png"))