import numpy as np
import scipy.stats as stats

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
                for j in range(len(sorted_values) - 1):
                    val = sorted_values[j]
                    idx = np.where(ys[i,:] >= val)
                    assign = (val - sorted_values[j+1])/len(idx[0])
                    probs[i, idx] += assign
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


def oau_entropy(orthop, n_classes):
    orthop_copy = np.copy(orthop)
    probs = np.zeros([n_classes, len(orthop)])
    i = 0
    for elem in orthop:
        for cl in elem:
            probs[cl, i] = 1.0
        i += 1

    tots = np.sum(probs, axis=1)
    indices = np.argsort(tots)[::-1]
    for i in indices:
        for j in range(len(orthop)):
            if i in orthop_copy[j]:
                orthop_copy[j] = [i]
    orthop_copy = np.array([t[0] for t in orthop_copy])
    probs = np.zeros([n_classes])
    for elem in orthop_copy:
        probs[elem] += 1.0
    probs /= sum(probs)
    return stats.entropy(probs, base=2)

def bet_entropy(orthop, n_classes):
        probs = np.zeros([n_classes])
        unif = np.ones([n_classes])
        for elem in orthop:
            for cl in elem:
                probs[cl] += 1.0/len(elem)
        probs /= sum(probs)
        unif = unif * (probs > 0)
        unif /= sum(unif)
        num =  stats.entropy(probs, base=2)
        den = stats.entropy(unif, base=2)
        h = 0 if num == 0 else num/den
        return h