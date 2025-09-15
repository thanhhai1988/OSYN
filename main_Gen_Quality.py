import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os
from data import create_GM_dataset, create_test_dataset
from preprocess import preprocess
from loss_zero_one import L01_set, L01_element
from KLDistance import df_kl_distance, sample_GMM
from utils import *
import argparse
import yaml
import ast


def get_args():
    parser = argparse.ArgumentParser(description="OSYN experiment arguments")
    #  file config.yaml
    parser.add_argument("--config", type=str, default="config.yaml",
                        help="Path to config file")

    # Define P0 distribution
    parser.add_argument("--means", type=ast.literal_eval,
                        default=[[0,0], [12, 15], [15, 6], [6,7], [3, 18]],
                        help="List of mean vectors for Gaussian components")
    parser.add_argument("--covs", type=ast.literal_eval,
                        default=[[[2, 0.5], [0.5, 4]],
                                 [[5, -2], [-2, 7]],
                                 [[1, 0.9], [0.9, 5]],
                                 [[10, -7], [-7, 15]],
                                 [[5, 0.9], [0.9, 5]]],
                        help="List of covariance matrices for Gaussian components")
    parser.add_argument("--weights", type=ast.literal_eval,
                        default=[1/20, 3/20, 4/20, 5/20, 7/20],
                        help="Mixture weights for Gaussian components")

    # Dataset sizes
    parser.add_argument("--train_size", type=int, default=5000)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--oracle_size", type=int, default=20000)

    # Random seed
    parser.add_argument("--seed", type=int, default=0)

    # Mean shift
    parser.add_argument("--a_s", type=ast.literal_eval,
                        default=[0, -0.25, -0.5, -0.75, -1],
                        help="List of scaling factors for Delta_mean")
    parser.add_argument("--Delta_mean", type=ast.literal_eval,
                        default=[[1, 0], [1, 0], [1, 0], [1, 0], [1, 0]],
                        help="Constant mean shift vectors")

    # OSYN parameters
    parser.add_argument("--g", type=int, default=50000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--N", type=int, default=50000)

    # Paths
    parser.add_argument("--save_path", type=str, default="/content/OSYN/Results/")
    parser.add_argument("--opt_data_path", type=str, default="/content/OSYN/Opt_data/")

    # Deltas
    parser.add_argument("--delta1", type=float, default=0.01)
    parser.add_argument("--delta2", type=float, default=0.2)

    # Parse CLI before
    cli_args, _ = parser.parse_known_args()

    # Load YAML config
    with open(cli_args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Pri CLI > YAML
    args_dict = vars(cli_args)
    for k, v in cfg.items():
        if args_dict.get(k) is None:
            args_dict[k] = v

    return argparse.Namespace(**args_dict)
 
def OSYN_Gen_Quality():
    args = get_args()
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.opt_data_path, exist_ok=True)
    
    D_train =  create_GM_dataset(args.means, args.covs, args.weights, args.train_size, args.seed)
    D_oracle = create_GM_dataset(args.means, args.covs, args.weights, args.oracle_size, seed = args.seed+10)
    D_test = create_test_dataset(args.means, args.covs, 0, args.test_size, seed = args.seed + 20)
    D_columns = D_oracle.columns
    df_results = pd.DataFrame(columns = ['a_scale', 'LB', 'Gap'])

    X_train, y_train = preprocess(D_train)
    X_small_test, y_small_test = preprocess(D_test, test_set = True)
    X_oracle, y_oracle = preprocess(D_oracle)
    
    name = 'Decision Tree'
    clf = DecisionTreeClassifier(random_state = 0)
    
    clf = clf.fit(X_train, y_train) # Pretrained model
    #KL distance
    df_distance = df_kl_distance(args.a_s, args.means, args.covs, args.weights, args.Delta_mean)
    # Check variance of random runs
    for a_value in args.a_s:
        LB = optim_per_Pg(a_value, args.g, args.T, args.N, args.means, args.covs, args.weights,
                 X_small_test, y_small_test, args.test_size, args.Delta_mean, args.opt_data_path,
                 args.save_path, clf, D_columns,
                 args.k, args.delta1, args.delta2)
        gap =  L01_set(X_oracle, y_oracle, clf) - LB
        print(f'a_scale = {a_value}: LB =  {LB}, Gap = {gap}')
        print('--------------------------------------')
        # Save results
        row = pd.DataFrame(data = [[a_value, LB, gap]], columns = ['a_scale', 'LB', 'Gap'])
        df_results = pd.concat([df_results, row], axis = 0)
        
if __name__ == "__main__":
        OSYN_Gen_Quality()
        
