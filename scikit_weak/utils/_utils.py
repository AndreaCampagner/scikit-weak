import numpy as np

def to_probs(ys):
    values = np.unique(np.add.reduce(ys))
    probs = np.zeros((len(ys),len(values)))
    for i in range(len(ys)):
        for val in ys[i]:
            idx = values[values == val]
            probs[i,idx] += 1
        probs[i,:] /= np.sum(probs[i,:])
    return probs

def prob_format(ys):
    if ys.ndim < 2:
        return False
    val_low = 1 - np.finfo(np.float32).eps
    val_upp = 1 + np.finfo(np.float32).eps
    vals = np.sum(ys, axis=1)
    if np.all(( vals >= val_low) & (vals <= val_upp) ):
        return True
    return False