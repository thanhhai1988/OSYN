import numpy as np

def L01_element(x, y, clf):
  # x: a feature vector of df
  # y: scalar (true label)
  # loss_element:  0_1 max = 1
  y_pred = clf.predict(x)
  return 1-np.sum(y_pred==y)

def L01_set(X, y, clf):
  y_pred = clf.predict(X)
  return 1-np.sum(y_pred==y)/X.shape[0]