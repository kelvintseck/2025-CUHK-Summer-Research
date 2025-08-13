import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import time



from joblib import Parallel, delayed
import time


import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")



from distributions import distParams, betaMix, generate_distribution
from estimators import gammaUpdate, betaEstimator, IS_selfNorm, IS_ET, CV_shift







def replicate_once(N, dist_index, delta, distParams, approach_prop, numFolds):
    dist_name, param = distParams[dist_index]
    samples = generate_distribution(dist_name, N, param)
    gamma = gammaUpdate(N, delta, samples, N)
    return betaEstimator(samples, gamma), approach_prop(samples, gamma, numFolds)


def mainLoop_parallel_joblib(generateDist, delta, Ns, approach_prop, numFolds_list, multiplier=100, n_jobs=None, timeout=None):
    num_dists = len(generateDist)
    results = [[list() for _ in range(num_dists)] for _ in range(len(Ns))]
    # Store results for each numFolds separately
    results_prop = [[[list() for _ in range(num_dists)] for _ in range(len(Ns))] for _ in range(len(numFolds_list))]

    startTime = time.time()

    for j, N in enumerate(Ns):
        print("=" * 100)
        print(f"Number of samples drawn: {N}")
        print(f"Number of rare samples: {N - 1 - int(N * (1 - N ** (delta - 1)))}")

        total_reps = int(Ns[-1] * multiplier)

        for nf_idx, numFolds in enumerate(numFolds_list):
            tasks = (delayed(replicate_once)(N, dist_index, delta, generateDist, approach_prop, numFolds)
                     for dist_index in range(num_dists)
                     for _ in range(total_reps))

            try:
                results_flat = Parallel(n_jobs=n_jobs, timeout=timeout)(tasks)
            except Exception as e:
                print(f"Timeout or error occurred: {e}")
                results_flat = []

            for idx, res in enumerate(results_flat):
                dist_index = idx // total_reps
                results[j][dist_index].append(res[0])
                results_prop[nf_idx][j][dist_index].append(res[1])

    print(f"Time usage: {time.time() - startTime:.2f}s")
    return results, results_prop


approach_idx = 0 
approaches = ["ETIS", "SNIS", "CV"]

def visualisation(results, results_prop, betaDist, Ns, distParams, numFolds_list):
    num_dists = len(betaDist)
    ncols = int(np.ceil(num_dists**(1/2)))
    nrows = int(np.ceil(num_dists / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 9), sharex=True)
    axes = np.array(axes).flatten()

    for i in range(num_dists):
        dist_biases_std = []
        dist_variances_std = []

        for j in range(len(Ns)):
            beta_estimates = np.array(results[j][i])
            dist_biases_std.append(np.nanmean(beta_estimates) - betaDist[i])
            dist_variances_std.append(np.nanvar(beta_estimates))

        ax_var = axes[i]
        ax_bias = ax_var.twinx()
        ax_bias.hlines(0, xmin=Ns[0], xmax=Ns[-1], color="r", linestyle=':')

        dist_name, param = distParams[i]
        ax_var.set_title(f"{dist_name} (β={betaDist[i]:.3f})")
        ax_var.set_yscale('log')

        ax_var.plot(Ns, dist_variances_std, color='blue', label='Var (Standard)')
        ax_bias.scatter(Ns, dist_biases_std, color='blue', label='Bias (Standard)')

        colors = plt.cm.tab10.colors
        for nf_idx, numFolds in enumerate(numFolds_list):
            dist_biases_prop = []
            dist_variances_prop = []
            for j in range(len(Ns)):
                beta_prop_estimates = np.array(results_prop[nf_idx][j][i])
                dist_biases_prop.append(np.nanmean(beta_prop_estimates) - betaDist[i])
                dist_variances_prop.append(np.nanvar(beta_prop_estimates))

            if numFolds != 1:
                ax_var.plot(Ns, dist_variances_prop, color=colors[nf_idx], linestyle='--', label=f'Var (Folds={numFolds})')
                ax_bias.scatter(Ns, dist_biases_prop, color=colors[nf_idx], label=f'Bias (Folds={numFolds})')
            else:
                ax_var.plot(Ns, dist_variances_prop, color=colors[nf_idx], linestyle='--', label=f'Var (Full)')
                ax_bias.scatter(Ns, dist_biases_prop, color=colors[nf_idx], label=f'Bias (Full)')

        if i == 0:
            lines_var, labels_var = ax_var.get_legend_handles_labels()
            lines_bias, labels_bias = ax_bias.get_legend_handles_labels()
            fig.legend(lines_var, labels_var, loc='lower center', ncol=5, frameon=False)
            fig.legend(lines_bias, labels_bias, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=5, frameon=False)

    for i in range(num_dists, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f'bias-var_{approaches[approach_idx]}.png')
    
    
    
    
    
    
    
    # Compute statistics
    stats_data = []
    for i in range(num_dists):
        dist_name, _ = distParams[i]
        row = {
            'Dist': dist_name,
            'beta': f"{betaDist[i]:.3f}"
        }
        
        # Variance reduction percentages
        for numFolds in numFolds_list:
            nf_idx = numFolds_list.index(numFolds)
            var_reds = []
            for j in range(len(Ns)):
                var_std = np.nanvar(np.array(results[j][i]))
                var_prop = np.nanvar(np.array(results_prop[nf_idx][j][i]))
                var_red = 100 * (var_std - var_prop) / var_std
                var_reds.append(var_red)
            label = 'Full' if numFolds == 1 else f'{numFolds}Folds'
            row[f'Var-Reduction% ({label})'] = np.nanmean(var_reds)

        
        beta_estimates_n0 = np.array(results[0][i])
        beta_estimates_n9 = np.array(results[-1][i])
        row[f'Bias (Standard, {Ns[0]})'] = np.nanmean(beta_estimates_n0) - betaDist[i]
        row[f'Bias (Standard, {Ns[-1]})'] = np.nanmean(beta_estimates_n9) - betaDist[i]
        
        for numFolds in numFolds_list:
            nf_idx = numFolds_list.index(numFolds)
            beta_prop_n0 = np.array(results_prop[nf_idx][0][i])
            beta_prop_n9 = np.array(results_prop[nf_idx][-1][i])
            label = 'Full' if numFolds == 1 else f'{numFolds}Folds'
            row[f'Bias ({label}, {Ns[0]})'] = np.nanmean(beta_prop_n0) - betaDist[i]
            row[f'Bias ({label}, {Ns[-1]})'] = np.nanmean(beta_prop_n9) - betaDist[i]
        
        stats_data.append(row)

    columns = ['Dist', 'beta', 'Var-Reduction% (Full)', 'Var-Reduction% (2Folds)', 'Var-Reduction% (5Folds)', 'Var-Reduction% (10Folds)',
               f'Bias (Standard, {Ns[0]})', f'Bias (Full, {Ns[0]})', f'Bias (2Folds, {Ns[0]})', f'Bias (5Folds, {Ns[0]})', f'Bias (10Folds, {Ns[0]})',
               f'Bias (Standard, {Ns[-1]})', f'Bias (Full, {Ns[-1]})', f'Bias (2Folds, {Ns[-1]})', f'Bias (5Folds, {Ns[0]})', f'Bias (10Folds, {Ns[-1]})']
    stats_df = pd.DataFrame(stats_data, columns=columns)
    print("\nSummary Statistics:")
    display(stats_df.round(4))
    

    stats_df.to_csv(f'summary_stats_{approaches[approach_idx]}.csv', index=False)




if __name__ == "__main__":
    np.random.seed(1155192082)
    generateDist = distParams
    betaDist = betaMix
    # delta δ is a hyperparameter
    delta = 0.75
    
    
    Ns = list(range(100, 1001, 100))
    # Ns = list(range(1000, 10001, 1000))
    numFolds_list = [1, 2, 5, 10]
    
    
    for approach_prop in [IS_ET, IS_selfNorm, CV_shift]:
        results, results_prop = mainLoop_parallel_joblib(generateDist, delta, Ns, approach_prop, numFolds_list, multiplier=50)
        visualisation(results, results_prop, betaMix, Ns, distParams, numFolds_list)
        approach_idx += 1