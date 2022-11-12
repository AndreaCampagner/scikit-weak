import numpy as np

def hellinger_distance(x, y):
  if x.shape != y.shape:
    raise ValueError("Different length")

  m1 = np.zeros(x.shape)
  m2 = np.zeros(y.shape)
  s1 = np.zeros((x.shape[0],x.shape[0]))
  s2 = np.zeros((y.shape[0],y.shape[0]))
  for c in range(x.shape[0]):
    m1[c] = x[c].mean
    m2[c] = y[c].mean
    s1[c,c] = x[c].std**2
    s2[c,c] = y[c].std**2


  s = (s1 + s2)/2
  res = np.exp(-1/8*(m1 - m2).dot(np.linalg.inv(s)).dot(m1 - m2))
  res *= np.sqrt(np.sqrt(np.linalg.det(s1))*np.sqrt(np.linalg.det(s2))/np.linalg.det(s))
  return np.sqrt(1 - res)


def mahalanobis_distance(x, y):
  if x.shape != y.shape:
    raise ValueError("Different length")

  m1 = np.zeros(x.shape)
  m2 = np.zeros(y.shape)
  s1 = np.zeros((x.shape[0],x.shape[0]))
  s2 = np.zeros((y.shape[0],y.shape[0]))
  for c in range(x.shape[0]):
    m1[c] = x[c].mean
    m2[c] = y[c].mean
    s1[c,c] = x[c].std**2
    s2[c,c] = y[c].std**2


  res = np.sqrt((m1 - m2).dot(np.linalg.inv(s1)).dot(m1-m2))
  res += np.sqrt((m1 - m2).dot(np.linalg.inv(s2)).dot(m1-m2))
  return res/2