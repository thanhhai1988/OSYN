import faiss
import numpy as np
import pandas as pd
import os
from collections import defaultdict
import json
from KLDistance import sample_GMM
from loss_zero_one import *
from preprocess import *
import warnings
warnings.filterwarnings("ignore")

def index_cluster(X, K):
    """
    Returns: K centroids, each is a sample in small test
    -------
    X: features of small set
    K: small set size
    """
    d = 2
    index = faiss.IndexFlatL2(d)
    centroids = []
    for i in range(K):
        s = X.iloc[[i]].values 
        centroids.append(s)
    centroids = np.array(centroids)
    centroids = centroids.reshape(K, d)
    index.add(centroids)
    
    return index

    
def check_area(s, index):
    """
    ----------
    Check area of sample features s with given index
    -------
    """
    I = index.search(s, 1)
    return I[1]

def generator_dis(Pg_means, Pg_covs, weights, index, test_size):
    """
    Returns distribution of generator on areas
    Pg = (Pg_1, ..., Pg_K)
    index: K centroids of K areas
    K = test size
    -------
    """
    Ng = np.zeros(test_size)
    Pg_dis = np.zeros(test_size)
    check_change = np.zeros(test_size)
    count_sum = 0
    T_dis = 20
    for no_iter in range(T_dis):
        N = 50000 # Tổng số pt sinh mỗi lần lặp
        # Thay doi Pg: 1) Pg = P_ori; Pg = P_1; Pg = P_2
        samples, labels = sample_GMM(N, Pg_means, Pg_covs, weights)
        for k in range(N):
            area = check_area(samples[k].reshape(1,-1), index).item() #int
            Ng[area] += 1
        count_sum += N
        Pg_dis = Ng/count_sum
        if (no_iter+1) == T_dis:
            print('Iteration ', no_iter+1, ':')
            print(f"Mean change:  {np.mean(np.abs(Pg_dis-check_change)):.6f}; Max change:  {np.max(np.abs(Pg_dis-check_change)):.6f}")
        check_change = Pg_dis.copy()
    return Pg_dis
    
def adjust_g(no_syn, k = 3):
  no_syn_adj = no_syn
  g = np.sum(no_syn_adj)
  K = len(no_syn_adj)
  for i in range(K):
    if (no_syn_adj[i]/g > 1/K):
      while (no_syn_adj[i]/g-1/K) >= k/K:
          no_syn_adj[i] -= 1
    elif (no_syn_adj[i]/g < 1/K):
      while ((- no_syn_adj[i]/g+1/K) >= k/K) or (no_syn_adj[i] == 0):
          no_syn_adj[i] += 1
  return no_syn_adj

def No_syn_adj(g, Pg_dis, k=3):
    """
    Returns: no of adjusted g_i synthetic points/area
    -------
    None.

    """
    no_syn_opt = np.random.multinomial(g, Pg_dis, 1)
    no_syn_opt = no_syn_opt[0]
    no_syn_adj = adjust_g(no_syn_opt, k)
    g_adj = np.sum(no_syn_adj)
    
    return no_syn_adj, g_adj

#Utils functions for OSYN algorithm
def check_no_synth_points(no_syn, no_syn_adj, test_size):
    """
    Check if no_syn >= no_syn_opt at each epoch
    no_syn: Total of synthetic points from epoch 0 -> current epoch
    K: size of small test set
    """
    check = (no_syn >= no_syn_adj)
    check = np.sum(check)
    return (check == test_size)

def shortage_dis(no_syn, no_syn_adj, test_size):
  """
  Turn distribution of shortage areas: how many shortage areas; /min/max/mean of insufficient no. of points
  """
  shortage = np.zeros(test_size)
  count = 0
  for i in range(test_size):
    if no_syn[i] < no_syn_adj[i]:
      shortage[i] = no_syn_adj[i] - no_syn[i]
      count += 1
  return count, count/test_size,  np.min(shortage), np.max(shortage), np.mean(shortage)


def Ng_adj_shortage(no_syn, no_syn_adj, test_size):
  """
  Turn no. of synth poitns g_i each area with shortage
  """
  g_adj_short = no_syn_adj.copy()
  for i in range(test_size):
    if (no_syn[i] < no_syn_adj[i]):
      g_adj_short[i] = no_syn[i]
  return g_adj_short


def check_point(L01_loss_area, g, Pg, no_syn_adj,  a_scale, 
                X, y, test_size, clf, 
                D_columns, opt_data_root, delta1 = 0.01, delta2 = 0.2):
    """
    Parameters
    ----------
    L01_loss_area : Losses of all synthetic elements to current step
    to compute a_hat, beta
    g : no. of target total synthetic points
    Pg : Distribution of generator each area
    no_syn_adj : No. of synthetic points/area
    a : scale of mean shift
    delta1 :  The default is 0.01.
    delta2 : The default is 0.2.

    Returns
    -------
    loss_syn : synthetic dataset loss
    epsilon : eps(G, S)
    uncer1 : B
    uncer2 : D
    lower_bound : RHS main inequality
    a_hat,C_h, beta : as in main theorem
    delta_lb : np.exp(-g*beta/(2*a_hat**2))
    cond : Check if delta_1 > delta_lb
    """
    # a_hat
    mean_loss_area = []
    for area_idx in range(test_size):
        mean_loss_area.append(np.mean(L01_loss_area[str(area_idx)]))

    a_hat =  np.max(mean_loss_area)

    # Compute beta
    beta = 0
    beta = 2*np.sum(np.multiply(np.array(mean_loss_area)**2, Pg))

    # Check delta condition
    cond = 0

    delta1_lb_L01 = np.exp(-g*beta/(2*a_hat**2))

    cond = np.float64(np.sum((delta1 > delta1_lb_L01)))
    delta_lb =   delta1_lb_L01

     # max loss C_h
    C_h = []
    for area_idx in range(test_size):
        C_h.append(np.max(L01_loss_area[str(area_idx)]))
    C_h = float(np.max(C_h))

    # F(G, h)
    df_optim = pd.DataFrame(columns = D_columns)

    for area_idx in range(test_size):
        thulai = pd.read_pickle(f'{opt_data_root}optim_data_a{a_scale}_area{area_idx}.pkl')
        df_optim= pd.concat([df_optim, thulai], axis = 0)

    X_optim = df_optim.iloc[:, :-1]
    y_optim = df_optim.iloc[:, -1]

    y_optim_label = LabelEncoder_Custom(y_optim.values) # Label Encoder: y = 0, 1, 2, 3, 4

    # Loss
    zero_one_loss_syn = L01_set(X_optim, y_optim_label, clf)
    loss_syn =  zero_one_loss_syn

    ########### Calculate Epsilon
    epsilon = 0
    for area_idx in range(test_size):
         df_area = pd.read_pickle(f'{opt_data_root}optim_data_a{a_scale}_area{area_idx}.pkl')

         X_syn =  df_area.iloc[:, :-1]
         y_syn =  df_area.iloc[:, -1]
         y_syn_label = LabelEncoder_Custom(y_syn.values) # Label Encoder: y = 0, 1, 2, 3, 4

         # Calulate epsilon value for each kind of loss for chung opt. data
         for i in range(df_area.shape[0]):
             epsilon_value_L01 = np.abs(L01_element(X_syn.iloc[[i]], y_syn_label[i],clf)
                          - L01_element(X.iloc[[area_idx]], y[area_idx], clf))
             epsilon += epsilon_value_L01


    epsilon = (epsilon/g).item()

    # Uncertainty
    uncertainty1 = 0
    for area_idx in range(test_size):
          uncertainty1 += no_syn_adj[area_idx]**2
    uncertainty1 = uncertainty1*np.log(1/delta2)/2/g**2 
    uncertainty1 = np.sqrt(uncertainty1)
    # C_h for L1 = 2, C_h = 1 for JS
    uncer1 = uncertainty1

    # Uncertainty2
    uncer2 = a_hat*np.log(1/delta1)/g
    # lower bound
    lower_bound = 0
    lower_bound =  (np.sqrt(loss_syn- epsilon - uncer1+ uncer2) - np.sqrt(uncer2))**2

    return loss_syn, epsilon, uncer1, uncer2, lower_bound, a_hat, C_h, beta, delta_lb,  cond

def check_point_shortage(L01_loss_area, g_adj_short, Pg_dis, Ng_adj_short,  a_scale,
                          X, y, test_size, clf, D_columns, opt_data_root, delta1 = 0.01, delta2 = 0.2):
    """
    Parameters
    ----------
    L01_loss_area : Losses of all synthetic elements to current step
    to compute a_hat, beta
    g : no. of target total synthetic points
    Pg : Distribution of generator each area
    no_syn_adj : No. of synthetic points/area
    a : scale of mean shift
    delta1 :  The default is 0.01.
    delta2 : The default is 0.2.

    Returns
    -------
    loss_syn : synthetic dataset loss
    epsilon : eps(G, S)
    uncer1 : B
    uncer2 : D
    lower_bound : RHS main inequality
    a_hat,C_h, beta : as in main theorem
    delta_lb : np.exp(-g*beta/(2*a_hat**2))
    cond : Check if delta_1 > delta_lb
    """
    mean_loss_area = np.zeros(test_size)
    for area_idx in range(test_size):
        if Ng_adj_short[area_idx] != 0:
            mean_loss_area[area_idx] = np.mean(L01_loss_area[str(area_idx)])
    a_hat =  np.max(mean_loss_area)

    # Compute beta
    beta = 2*np.sum(np.multiply(np.array(mean_loss_area)**2, Pg_dis))

    # Check delta condition
    delta1_lb = np.exp(-g_adj_short*beta/(2*a_hat**2))

    cond = np.float64(np.sum((delta1 > delta1_lb)))

    #max loss C_h
    C_h = []
    for area_idx in range(test_size):
        if Ng_adj_short[area_idx] != 0:
            C_h.append(np.max(L01_loss_area[str(area_idx)]))
        else:
            C_h.append(0) # Avoid errors when there aren't syn. points in this area
    C_h = float(np.max(C_h))

    #F(G, h)
    df_optim = pd.DataFrame(columns = D_columns)

    for area_idx in range(test_size):
        thulai = pd.read_pickle(f'{opt_data_root}optim_data_a{a_scale}_area{area_idx}.pkl')
        df_optim= pd.concat([df_optim, thulai], axis = 0)

    X_optim = df_optim.iloc[:, :-1]
    y_optim = df_optim.iloc[:, -1]
    y_optim_label = LabelEncoder_Custom(y_optim.values) # Label Encoder: y = 0, 1, 2, 3, 4

    # Loss
    zero_one_loss_syn = L01_set(X_optim, y_optim_label, clf)
    loss_syn =  zero_one_loss_syn

    ########### Calculate Epsilon
    epsilon = 0
    for area_idx in range(test_size):
        if Ng_adj_short[area_idx] != 0:
            df_area = pd.read_pickle(f'{opt_data_root}optim_data_a{a_scale}_area{area_idx}.pkl')

            X_syn =  df_area.iloc[:, :-1]
            y_syn =  df_area.iloc[:, -1]
            y_syn_label = LabelEncoder_Custom(y_syn.values) # Label Encoder: y = 0, 1, 2, 3, 4

            # Calulate epsilon value for each kind of loss for chung opt. data
            for i in range(df_area.shape[0]):
                epsilon_value_L01 = np.abs(L01_element(X_syn.iloc[[i]], y_syn_label[i],clf)
                          - L01_element(X.iloc[[area_idx]], y[area_idx], clf))
                epsilon += epsilon_value_L01


    epsilon = (epsilon/g_adj_short).item()

    # Uncertainty
    uncertainty1 = 0
    for area_idx in range(test_size):
        uncertainty1 += Ng_adj_short[area_idx]**2
    uncertainty1 = uncertainty1*np.log(1/delta2)/2/g_adj_short**2
    uncertainty1 = np.sqrt(uncertainty1)
    # C_h for L1 = 2, C_h = 1 for JS
    uncer1 = uncertainty1

    # Uncertainty2

    uncer2 = a_hat*np.log(1/delta1)/g_adj_short


    # lower bound
    lower_bound =  (np.sqrt(loss_syn- epsilon - uncer1+ uncer2) - np.sqrt(uncer2))**2

    return loss_syn, epsilon, uncer1, uncer2, lower_bound, a_hat, C_h, beta, delta1_lb,  cond



def convert_to_df(samples, labels):
  """
  Convert synthetic features and labels to dataframe type
  """
  df_gen = pd.DataFrame(samples, columns = ['x1', 'x2'])
  df_gen['label'] = labels
  return df_gen


# Luu results duoi 2 dinh dang csv
def save_results(save_root, file, a_scale, no_iter):
  da = [file['loss_syn'], file['epsilon'], file['uncertainty1'], file['uncertainty2'], file['lower_bound'],
         file['a_hat'],file['C_h'], file['beta'], file['delta_lb'], file['cond']]
  da = np.array(da)
  da = da.reshape(1, 10)
  df_results = pd.DataFrame(data = da, 
                            columns = ['loss_syn', 'epsilon', 'uncertainty1', 'uncertainty2', 
                                       'lower_bound', 'a_hat', 'C_h', 'beta', 'delta_lb', 'cond'])
  df_results.to_csv(f'{save_root}results_a{a_scale}_iter{no_iter+1}.csv', index=False)



def optim_per_Pg(a_scale, g, T, N, means, covs, weights,
                 X, y, test_size, Delta_mean, opt_data_root,
                 save_root, clf, D_columns, 
                 k = 3, delta1 = 0.01, delta2 = 0.2):
  """
  a:Scale of mean shift
  k: cluster radius for optimize synthetic points
  """
  # Make dir save optim synthetic points: rewrite for new a 
  if not os.path.exists(opt_data_root):
    os.mkdir(opt_data_root)
  # Calculate corressponding means, covariances of corresponding Pg
  means_changed = np.add(means, a_scale*np.array(Delta_mean))
  
  # Compute index centroids 
  index = index_cluster(X, test_size)

  # Find P_g on K areas
  print('Section 1: Find distribution of Pg')
  Pg_dis = generator_dis(means_changed, covs, weights,
                         index, test_size)
  print('-------------End of Section 1 -----------------------')

  # Find num_syn_opt, num_syn_adj of K areas
  no_syn_adj, g_adj = No_syn_adj(g, Pg_dis, k)
  print('Number of adjusted synthetic points: ', g_adj)

  # Create K opt. data sets
  df_gen = pd.DataFrame(columns = D_columns)
  for area_idx in range(test_size):
    df_gen.to_pickle(f'{opt_data_root}optim_data_a{a_scale}_area{area_idx}.pkl')

  # Make loss area
  L01_loss_area = defaultdict(lambda: list())
  for area_idx in range(test_size):
    L01_loss_area[str(area_idx)] = []

  # Optimize
  no_syn = np.zeros(test_size)
  epoch_suff = 0
  print('Section 2: Find optimized synthetic points and Lower Bound Value')
  # Find Pg
  for no_iter in range(T):
    samples, labels = sample_GMM(N,means_changed, covs, weights)

    # convert to df
    df_gen_overall = convert_to_df(samples, labels)

    # Divide into areas: each contains indexes
    data = defaultdict(lambda: list())
    for i in range(N):
      data[str(check_area(samples[i].reshape(1,-1), index).item())].append(i)

    # Optimize each area independently
    for area_idx in range(test_size):
      if str(area_idx) in data.keys():
       df_gen = pd.read_pickle(f'{opt_data_root}optim_data_a{a_scale}_area{area_idx}.pkl')

       # Loc du lieu sinh thuoc vung tu tong N pt duoc sinh
       df_area = df_gen_overall.iloc[data[str(area_idx)]]

       # Cập nhật n
       n = df_area.shape[0] # Số phần tử vừa sinh Ở bước hiện tại thuộc vùng

       X_syn =  df_area.iloc[:, :-1]
       y_syn =  df_area.iloc[:, -1]
       y_syn_label = LabelEncoder_Custom(y_syn.values) # Label Encoder: y = 0, 1, 2, 3, 4

       # Cập nhật g
       no_syn[area_idx] = no_syn[area_idx] + n # Tổng số phần tử đã sinh thuộc vùng ĐẾN bước hiện tại
       # Cập nhật Dãy loss của các vùng
       for j in range(n):
            L01_loss_area[str(area_idx)].append(L01_element(X_syn.iloc[[j]], y_syn_label[j], clf))
       # Cập nhật tập tìm kiếm điểm tối ưu

       search_df = pd.concat([df_gen, df_area], axis = 0) # Hợp của tập sinh tốt ở bước trước và tập vừa sinh ở bước hiện tại

       # Tìm no_syn_opt[area] điểm sinh tối ưu
       targets = []
       X_search =  search_df.iloc[:, :-1]
       y_search =  search_df.iloc[:, -1]       
       y_search_label = LabelEncoder_Custom(y_search.values) # Label Encoder: y = 0, 1, 2, 3, 4

       for i in range(search_df.shape[0]):
          target_L01 = -(L01_element(X_search.iloc[[i]], y_search_label[i], clf)- np.abs(L01_element(X.iloc[[area_idx]], y[area_idx], clf)
                                                                              - L01_element(X_search.iloc[[i]], y_search_label[i], clf)))
          targets.append(target_L01)

       targets = np.array(targets)
       targets = targets.reshape(-1)

       # Get the indices of the k smallest elements
       if search_df.shape[0] <= no_syn_adj[area_idx]: # Nếu số phần tử thuộc vùng sinh nhỏ hơn g_optim thì lấy toàn bộ làm tập tối ưu
             df_gen = search_df
       else:
             idx_min = np.argpartition(targets, no_syn_adj[area_idx])[:no_syn_adj[area_idx]]

             # Cập nhật tập tốt - dạng bảng, gồm cả label
             df_gen = search_df.iloc[idx_min]

       # Lưu
       df_gen.to_pickle(f'{opt_data_root}optim_data_a{a_scale}_area{area_idx}.pkl')

    print('Iteration', no_iter+1, '/', T)
    if  epoch_suff == 0:
      if  check_no_synth_points(no_syn, no_syn_adj, test_size):
          epoch_suff = no_iter
          # Lưu checkpoints
          check = check_point(L01_loss_area, g_adj, Pg_dis, no_syn_adj,  a_scale, 
                              X, y, test_size, clf, D_columns, opt_data_root)
          results = {
                    'loss_syn': check[0],
                    'epsilon': check[1],
                    'uncertainty1': check[2],
                    'uncertainty2': check[3],
                    'lower_bound': check[4],
                    'a_hat': check[5],
                    'C_h': check[6],
                    'beta': check[7],
                    'delta_lb': check[8],
                    'cond': check[9]
                    }
          print(results)
          save_results(save_root, results, a_scale, no_iter)
      else:
        print(shortage_dis(no_syn, no_syn_adj, test_size))
        Ng_adj_short = Ng_adj_shortage(no_syn, no_syn_adj, test_size)
        g_adj_short = np.sum(Ng_adj_short)
        check = check_point_shortage(L01_loss_area, g_adj_short, Pg_dis, Ng_adj_short,  a_scale,
                                 X, y, test_size, clf, D_columns, opt_data_root, delta1, delta2)
        results = {
                    'loss_syn': check[0],
                    'epsilon': check[1],
                    'uncertainty1': check[2],
                    'uncertainty2': check[3],
                    'lower_bound': check[4],
                    'a_hat': check[5],
                    'C_h': check[6],
                    'beta': check[7],
                    'delta_lb': check[8],
                    'cond': check[9]
                    }
        print(results)
        save_results(save_root, results, a_scale, no_iter)
    else:
      # Save checkpoints
      check = check_point(L01_loss_area, g_adj, Pg_dis, no_syn_adj,  a_scale, X, y, test_size, clf, 
                       D_columns, opt_data_root, delta1, delta2)
      results = {
        'loss_syn': check[0],
        'epsilon': check[1],
        'uncertainty1': check[2],
        'uncertainty2': check[3],
        'lower_bound': check[4],
        'a_hat': check[5],
        'C_h': check[6],
        'beta': check[7],
        'delta_lb': check[8],
        'cond': check[9]
      }
      print(results)
      save_results(save_root, results, a_scale, no_iter)
  return results['lower_bound']


















