import numpy as np

def to_probs(ys, uniform=False):
    if ys.ndim < 2:
        values = np.unique(np.add.reduce(ys))
        probs = np.zeros((len(ys),len(values)))
        for i in range(len(ys)):
            for val in ys[i]:
                idx = values[values == val]
                probs[i,idx] += 1
            probs[i,:] /= np.sum(probs[i,:])
        return probs
    else:
        probs = np.zeros(ys.shape)
        if uniform:
            probs = ys/np.sum(ys, axis=1)[:, np.newaxis]
        else:
            for i in range(ys.shape[0]):
                values = np.unique(ys[i])
                sorted_values = np.sort(values)[::-1]
                if sorted_values[-1] != 0.0:
                    sorted_values = np. append(sorted_values, 0.0)
                it = np.nditer(sorted_values, flags=['f_index'], order="K")
                for val in it:
                    if it.index == len(sorted_values) - 1:
                        break
                    idx = np.where(ys[i,:] >= val)
                    probs[i, idx[0]] = (val - sorted_values[it.index+1])/len(idx[0])
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