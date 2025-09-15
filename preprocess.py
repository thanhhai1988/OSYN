import numpy as np
from sklearn.preprocessing import LabelEncoder

def LabelEncoder_Custom(y):
  """
  Encoder for biased small test target
  """
  encoded_y = np.zeros(len(y))
  for i in range(len(y)):
    if y[i] == 0:
      encoded_y[i] = 0
    elif y[i] == 1:
      encoded_y[i] = 1
    elif y[i] == 2:
      encoded_y[i] = 2
    elif y[i] == 3:
      encoded_y[i] = 3
    else:
      encoded_y[i] = 4
  encoded_y = encoded_y.astype(int)
  return encoded_y

def preprocess(D, test_set = False):
  """
  Preprocess dataframe D into:
  X: df of features
  y: label encoded vector of class
  """
  X = D.drop('label', axis = 1)
  y = D['label']

  if test_set:
    y = LabelEncoder_Custom(y)
  else: 
    y = LabelEncoder().fit_transform(y)
  return X, y