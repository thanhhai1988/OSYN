import numpy as np
import pandas as pd

def create_GM_dataset(
        means, covs, weights, size, seed = 0):
    """

    Samples datasets (train/test/oracle) from means, covs, weights, size
    Output: a dataframe consisting of features and labels of size = size
    """
    #Number of GM samples/cluster
    np.random.seed(seed)
    no_samples =  np.random.multinomial(size, weights)
    # Generate GM samples/cluster
    samples = []
    for i in range(len(means)):   
        x = np.random.multivariate_normal(means[i], 
                                          covs[i], 
                                          no_samples[i])
        samples.append(x)
    X = np.vstack(samples)
    
    labels = np.repeat(np.arange(len(no_samples)), no_samples)
    
    df = pd.DataFrame(X, columns = ['x1', 'x2'])
    df['label']= labels
    
    return df

def create_test_dataset(means, covs, class_id, size, seed = 0):
    """
    Create a biased small test belonging to one cluster class_id
    of Gausssian Mixture of size = size
    ----------
    Output: a dataframe of 3 columns: [x1,x2, class_id]
    -------
    """
    np.random.seed(seed)
    X = np.random.multivariate_normal(means[class_id], 
                                      covs[class_id], 
                                      size)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['label'] = class_id
    
    return df

    

