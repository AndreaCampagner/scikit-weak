from sklearn.metrics.pairwise import pairwise_kernels
import scipy.stats as stats
import numpy as np

def mean_embedding_kernel(x, y, kernel='linear', n_samples=1000, gamma=1):
  if (type(x[0].distrib) == stats._continuous_distns.norm_gen) and (type(y[0].distrib) == stats._continuous_distns.norm_gen):
    return rbf_embedding_kernel(x, y, gamma)
  res = 0
  X = np.zeros((n_samples, x.shape[0]))
  Y = np.zeros((n_samples, y.shape[0]))
  for i in range(n_samples):
    for c in range(x.shape[0]):
      X[i,c] = x[c].sample_value()
      Y[i,c] = y[c].sample_value()
  
  K = pairwise_kernels(X, Y, metric=kernel)

  return np.sum(K)/(n_samples**2)


def rbf_embedding_kernel(x, y, gamma=1):
  if x.shape != y.shape:
    raise ValueError("Different length")

  m1 = np.zeros(x.shape)
  m2 = np.zeros(y.shape)
  s1 = np.zeros((x.shape[0],x.shape[0]))
  s2 = np.zeros((y.shape[0],y.shape[0]))
  for c in range(x.shape[0]):
    m1[c] = x[c].distrib.mean()
    m2[c] = y[c].distrib.mean()
    s1[c,c] = x[c].distrib.std()**2
    s2[c,c] = y[c].distrib.std()**2

  res = np.exp(-0.5*(m1 - m2).dot(np.linalg.inv(s1 + s2 + 1/gamma*np.eye(x.shape[0]))).dot(m1-m2))
  res /= np.sqrt(np.linalg.det(gamma*s1 + gamma*s2 + np.eye(x.shape[0])))
  return res