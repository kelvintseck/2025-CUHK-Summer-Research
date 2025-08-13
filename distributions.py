import numpy as np
from scipy.special import gamma as gammaFunc


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")




# Pareto Distribution (Type I)
def generatePareto(m, i):
    kappa = 1/(0.2 + 0.025 * i)
    tau = 1 - 1/kappa
    u = np.random.uniform(0, 1, m)
    samples = tau * (1 - u) ** (-1/kappa)
    return samples
# tail indices = kappa^{-1}

# Student's t Distribution
def generateT(m, i):
    omega = 1/(0.25 + 0.05 * i)
    nu = (np.pi * omega) ** (-1/2) * gammaFunc((omega+1)/2) / gammaFunc(omega/2)
    z = np.random.standard_t(omega, m)
    samples = np.abs(z) + 3 - np.mean(np.abs(z))
    return samples
# tail indices = omega^{-1}

# Fréchet Distribution
def generateFrechet(m, i):
    alpha = 1/(0.225 + 0.025 * i)
    s = 1/gammaFunc(1 - 1/alpha)
    u = np.random.uniform(0, 1, m)
    samples = s * (-np.log(u)) ** (-1/alpha)
    return samples
# tail indices = alpha^{-1}


distPareto = [lambda m, idx=i: generatePareto(m, idx) for i in range(1, 11)]
distT = [lambda m, idx=i: generateT(m, idx) for i in range(1, 11)]
distFrechet = [lambda m, idx=i: generateFrechet(m, idx) for i in range(1, 11)]
# i = 1 is optimal in all cases.
betaPareto = [(0.2 + 0.025 * i) for i in range(1, 11)]
betaT = [(0.25 + 0.05 * i) for i in range(1, 11)]
betaFrechet = [(0.225 + 0.025 * i) for i in range(1, 11)]


# Mixture of Pareto, T, Frechet

distParams = [
    ("Pareto", 1),
    ("Pareto", 5),
    ("Pareto", 10),
    ("Student's t", 1),
    ("Student's t", 5),
    ("Student's t", 10),
    ("Fréchet", 1),
    ("Fréchet", 5),
    ("Fréchet", 10)
]

betaMix = [0.225, 0.325, 0.45, 0.3, 0.5, 0.75, 0.25, 0.35, 0.475]



def generate_distribution(name, m, param):
    if name == "Pareto":
        return generatePareto(m, param)
    elif name == "Student's t":
        return generateT(m, param)
    elif name == "Fréchet":
        return generateFrechet(m, param)
