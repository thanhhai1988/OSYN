import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

def sample_GMM(size, means, covs, weights):
  # size: no. of syntetic points
  # means, covs: list of mean vectors, cov matrices
  # weights: uniform or non-uniform
  #            : uniform_GMM = [0.2,0.2, 0.2, 0.2, 0.2]
  # non uniform: weight_GMM = [1/20, 3/20, 4/20, 5/20, 7/20]
  components = np.random.multinomial(size, weights)
  sample1 = np.random.multivariate_normal(means[0], covs[0], components[0])
  sample2 = np.random.multivariate_normal(means[1], covs[1], components[1])
  sample3 = np.random.multivariate_normal(means[2], covs[2], components[2])
  sample4 = np.random.multivariate_normal(means[3], covs[3], components[3])
  sample5 = np.random.multivariate_normal(means[4], covs[4], components[4])
  labels = np.array([0] * components[0]+
                    [1] * components[1] +
                    [2] * components[2] +
                    [3] * components[3] +
                    [4] * components[4])
  # Combine 5 samples
  samples_combined= np.concatenate((sample1, sample2, sample3, sample4, sample5), axis=0)
  return samples_combined, labels

def gmm_pdf(x, weights, means, covariances):

    K = len(weights)
    pdf = 0.0

    for k in range(K):
        # Calculate the PDF for each Gaussian component
        component_pdf = weights[k] * multivariate_normal.pdf(x, mean=means[k], cov=covariances[k])
        pdf += component_pdf

    return pdf

def gmm_kl(gmm_p, gmm_q, weights, n_samples=1e7):
    X, _ = sample_GMM(n_samples, gmm_p[0], gmm_p[1], weights)
    p_X = (gmm_pdf(X, gmm_p[2], gmm_p[0], gmm_p[1]))
    q_X = (gmm_pdf(X, gmm_q[2], gmm_q[0], gmm_q[1]))
    return np.mean(np.log(p_X/q_X))

def df_kl_distance(a_s, means, covs, weights, Delta_mean):
    """
    Create a dataframe containing Kl distance 
    between shifted distribution Pg and original dis. P0
    """
    df_distance = pd.DataFrame(columns=['a_means_change', 'KL(Pg||P0)'])
    gmm_ori = [means, covs, weights]
    for a_value in a_s:
      means_changed = np.add(means, a_value*np.array(Delta_mean))
      gmm_changed = [means_changed, covs, weights]
      kl = gmm_kl(gmm_changed, gmm_ori, weights)
      # New rows to add
      new_row = pd.DataFrame([{'a_means_change': a_value, 'KL(Pg||P0)': kl}])

      # Concatenate the new rows
      df_distance = pd.concat([df_distance, new_row], ignore_index=True)
    
    return df_distance