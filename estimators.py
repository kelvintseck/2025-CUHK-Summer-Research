import numpy as np
from scipy.optimize import minimize, fsolve, brentq
from scipy.special import gamma as gammaFunc
from scipy.stats import rankdata
import warnings


import matplotlib.pyplot as plt
from distributions import generatePareto




def gammaUpdate(T: int, delta: float, samples: np.ndarray, N: int) -> float:
    target = int(N * (1 - T ** (delta - 1)))
    if target >= N:
        return np.min(samples)
    if target < 0:
        return np.max(samples)
    return np.partition(samples, target)[target]  # Return the largest gamma


def betaEstimator(samples: np.ndarray, gamma: float) -> float:
    validSamples = samples[samples > gamma]
    return np.mean(np.log(validSamples / gamma)) if len(validSamples) > 0 else 0.0


# ================================================================================================================================ #

def CV_shift(samples: np.ndarray, gamma: float, numFolds: int) -> float:
    try:
        validSamples = samples[samples > gamma]
        X = np.log(validSamples / gamma)
        n = len(X)
        if n < numFolds:
            return np.nan

        if numFolds == 1:
            # Full dataset estimation
            beta_hat = np.mean(X)
            ranks = np.argsort(np.argsort(X))
            U = (ranks + 0.5) / (n + 1.0)       
            Y = -beta_hat * np.log(1.0 - U)     
            
            # Rescale and shift Y to match X scale
            Xc = X - np.mean(X)
            Yc = Y - np.mean(Y)
            cov = np.mean(Xc * Yc)
            varY = np.var(Yc)
            alpha = cov / varY if varY > 0 else 0.0
            delta = np.mean(X) - alpha * np.mean(Y)
            Y_rescaled = alpha * Y + delta
            
            # Compute b_opt
            Xc_rescaled = X - np.mean(X)
            Yc_rescaled = Y_rescaled - np.mean(Y_rescaled)
            cov_rescaled = np.mean(Xc_rescaled * Yc_rescaled)
            varY_rescaled = np.var(Yc_rescaled)
            b_opt = cov_rescaled / varY_rescaled if varY_rescaled > 0 else 0.0
            
            # Control variate estimate
            mu_hat = np.mean(X - b_opt * (Y_rescaled - beta_hat))
            
        
        else:
            # K-fold cross-validation
            idx = np.arange(n)
            folds_idx = np.array_split(idx, numFolds)
            estimates = []

            for k in range(numFolds):
                val_idx = folds_idx[k]
                train_idx = np.concatenate([folds_idx[j] for j in range(numFolds) if j != k])
                X_train = X[train_idx]
                X_val = X[val_idx]

                # Estimate beta_hat from training mean
                beta_hat = np.mean(X_train)

                # Sort training set for ECDF
                sort_train_idx = np.argsort(X_train)
                X_train_sorted = X_train[sort_train_idx]

                # U values for training set (original order)
                ranks_train = np.argsort(np.argsort(X_train))
                U_train = (ranks_train + 0.5) / (len(X_train) + 1.0)
                Y_train = -beta_hat * np.log(1.0 - U_train)

                # U values for validation set: rank within training set
                ranks_in_train = np.searchsorted(X_train_sorted, X_val, side='right')
                U_val = (ranks_in_train + 0.5) / (len(X_train) + 1.0)
                Y_val = -beta_hat * np.log(1.0 - U_val)

                # Rescale Y_train to match X_train scale
                Xc = X_train - np.mean(X_train)
                Yc = Y_train - np.mean(Y_train)
                cov_train = np.mean(Xc * Yc)
                varY_train = np.var(Yc)
                alpha = cov_train / varY_train if varY_train > 0 else 0.0
                delta = np.mean(X_train) - alpha * np.mean(Y_train)
                Y_train_rescaled = alpha * Y_train + delta

                # Apply same rescaling to validation Y
                Y_val_rescaled = alpha * Y_val + delta

                # Compute b_opt using rescaled training data
                Xc_rescaled = X_train - np.mean(X_train)
                Yc_rescaled = Y_train_rescaled - np.mean(Y_train_rescaled)
                cov_rescaled = np.mean(Xc_rescaled * Yc_rescaled)
                varY_rescaled = np.var(Yc_rescaled)
                b_opt = cov_rescaled / varY_rescaled if varY_rescaled > 0 else 0.0
                

                # Control variate estimate on validation fold
                est_k = np.mean(X_val - b_opt * (Y_val_rescaled - beta_hat))
                estimates.append(est_k)
                
                # print(b_opt, np.corrcoef(X_val, Y_val))
            

            mu_hat = float(np.mean(estimates))

        return mu_hat

    except Exception:
        return np.nan
    
    
    
# Shifting is better
    


def CV(samples: np.ndarray, gamma: float, numFolds: int) -> float:
    def compute_Y(X_input, beta_hat):
        ranks = np.argsort(np.argsort(X_input))
        U = (ranks + 0.5) / (len(X_input) + 1.0)
        return -beta_hat * np.log(1.0 - U)
    
    try:
        validSamples = samples[samples > gamma]
        X = np.log(validSamples / gamma)
        n = len(X)
        if n < numFolds:
            return np.nan
        
        if numFolds == 1:
            # Full dataset
            beta_hat = np.mean(X)
            Y = compute_Y(X, beta_hat)

            # Compute b_opt
            Xc = X - np.mean(X)
            Yc = Y - np.mean(Y)
            b_opt = np.mean(Xc * Yc) / np.var(Yc)
            b_opt = 1

            # Control variate estimate
            mu_hat = np.mean(X - b_opt * (Y - beta_hat))

        else:
            # K-fold cross-validation
            idx = np.arange(n)
            folds_idx = np.array_split(idx, numFolds)
            estimates = []

            for k in range(numFolds):
                val_idx = folds_idx[k]
                train_idx = np.concatenate([folds_idx[j] for j in range(numFolds) if j != k])
                
                X_train = X[train_idx]
                X_val = X[val_idx]

                # Estimate beta_hat from training mean
                beta_hat = np.mean(X_train)

                # Compute Y for train and val sets
                Y_train = compute_Y(X_train, beta_hat)

                # For validation set: rank within training set
                sort_train_idx = np.argsort(X_train)
                X_train_sorted = X_train[sort_train_idx]
                ranks_in_train = np.searchsorted(X_train_sorted, X_val, side='right')
                U_val = (ranks_in_train + 0.5) / (len(X_train) + 1.0)
                Y_val = -beta_hat * np.log(1.0 - U_val)

                # Compute b_opt from training set
                Xc_train = X_train - np.mean(X_train)
                Yc_train = Y_train - np.mean(Y_train)
                b_opt = np.mean(Xc_train * Yc_train) / np.var(Yc_train)
                b_opt = 1

                # Control variate estimate on validation fold
                est_k = np.mean(X_val - b_opt * (Y_val - beta_hat))
                estimates.append(est_k)
                
                # print(b_opt, np.corrcoef(X_val, Y_val))
            
            mu_hat = float(np.mean(estimates))

        return mu_hat

    except Exception:
        return np.nan


# ================================================================================================================================ #


def IS_selfNorm(samples: np.ndarray, gamma: float, numFolds: int) -> float:
    try:
        validSamples = samples[samples > gamma]
        X = np.log(validSamples / gamma)
        n = len(X)
        if n < numFolds:
            return np.nan

        if numFolds == 1:
            # Full dataset estimation
            beta_hat = np.mean(X)
            lam_hat = 1.0 / beta_hat
            theta = lam_hat / 2.0
            w = np.exp(-theta * X)
            w /= np.sum(w)
            mu_hat = np.sum(w * X)
        else:
            # K-fold cross-validation
            idx = np.arange(n)
            folds_idx = np.array_split(idx, numFolds)
            estimates = []
            
            for k in range(numFolds):
                val_idx = folds_idx[k]
                train_idx = np.concatenate([folds_idx[j] for j in range(numFolds) if j != k])
                X_train = X[train_idx]
                X_val = X[val_idx]
                
                beta_hat_train = np.mean(X_train)
                lam_hat_train = 1.0 / beta_hat_train
                theta_train = lam_hat_train / 2.0
                w_val = np.exp(-theta_train * X_val)
                w_val /= np.sum(w_val)
                est_k = np.sum(w_val * X_val)
                estimates.append(est_k)
            
            mu_hat = np.mean(estimates)
        
        return float(mu_hat)
    
    except Exception:
        return np.nan


# ================================================================================================================================ #

def IS_ET(samples: np.ndarray, gamma: float, numFolds: int) -> float:
    try:
        validSamples = samples[samples > gamma]
        X = np.log(validSamples / gamma)
        n = len(X)
        if n < numFolds:
            return np.nan

        if numFolds == 1:
            # Full dataset estimation
            beta_hat = np.mean(X)
            lambda_hat = 1.0 / beta_hat
            theta = lambda_hat / 2.0
            psi_hat = -np.log(1.0 - theta / lambda_hat)
            mu_hat = np.mean(X * np.exp(-theta * X + psi_hat))
        else:
            # K-fold cross-validation
            idx = np.arange(n)
            folds_idx = np.array_split(idx, numFolds)
            estimates = []

            for k in range(numFolds):
                val_idx = folds_idx[k]
                train_idx = np.concatenate([folds_idx[j] for j in range(numFolds) if j != k])
                X_train = X[train_idx]
                X_val = X[val_idx]

                # Estimate lambda_hat from training set
                beta_hat = np.mean(X_train)
                lambda_hat = 1.0 / beta_hat
                theta = lambda_hat / 2.0
                psi_hat = -np.log(1.0 - theta / lambda_hat)

                # IS estimate on validation set
                est_k = np.mean(X_val * np.exp(-theta * X_val + psi_hat))
                estimates.append(est_k)

            mu_hat = np.mean(estimates)

        return float(mu_hat)

    except Exception:
        return np.nan



if __name__ == "__main__":
    samples = generatePareto(10000, 5)
    CV_shift(samples, gammaUpdate(10000, 0.8, samples, 10000), 10)












