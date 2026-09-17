import numpy as np
import warnings

def lee_valiant_estimator(samples, num_trim, iter=100):
    """
    Inputs:
    samples: 1-d list or numpy array of n samples.
    num_trim (int or float): the total amount of downweighted samples -> set it to however many samples you would want to trim.
                            For small failure probability delta, set it to 1/3 ln(2/delta)
    iter: number of iterations for refining the mean estimate, shouldn't matter *too* much.
    
    Please cite:
    - Jasper C.H. Lee and Paul Valiant. Optimal sub-gaussian mean estimation in R. In Proc. FOCS'21, pages 672–683. 2022 (Note, pandemic)
    - Jasper C. H. Lee, Walter McKelvie, Maoyuan Song, and Paul Valiant.
      All-purpose mean estimation over R: Optimal sub-gaussianity with outlier robustness and low moments performance. In Proc. ICML’25, 2025.
    """
    samples = np.asarray(samples, dtype=float)
    n = len(samples)
    
    # Edge case when all samples are identical
    if np.max(samples) == np.min(samples):
        return samples[0]
        
    # Median-of-means, with random shuffling just in case the samples are strangely ordered
    samples_shuffled = np.random.permutation(samples)
    
    num_groups = max(1, int(np.floor(num_trim*3)))
    groups = np.array_split(samples_shuffled, num_groups)
    group_means = [np.mean(g) for g in groups]
    
    kappa = np.median(group_means)
    
    # Lee-Valiant Estimator, iterative version
    numerator = (num_trim) - np.arange(n - 1, -1, -1)
    
    for _ in range(iter):
        dev = samples - kappa
        sqdist = dev**2
        sqdist_sorted = np.sort(sqdist)
        
        denominator = np.cumsum(sqdist_sorted)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            alphas_cand = numerator / denominator
            
        sqdist_shifted = np.append(sqdist_sorted[1:], np.inf)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            cond1 = (alphas_cand * sqdist_shifted) >= 1
            cond2 = (alphas_cand * sqdist_sorted) <= 1
        
        mask = cond1 & cond2
        
        if not np.any(mask):
            mask[-1] = True
            
        alpha = alphas_cand[mask][0]
        
        est = kappa + (1 / n) * np.sum(dev * (1 - np.minimum(alpha * dev**2, 1)))
        kappa = est
        
    return est