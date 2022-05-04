import tensorflow as tf

@tf.function
def fuzzy_cross_entropy(y_true, y_pred):
    return fuzzy_loss(y_true, y_pred,
                      tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE))
@tf.function
def fuzzy_hinge(y_true, y_pred):
    return fuzzy_loss(y_true, y_pred,
                      tf.keras.losses.CategoricalHinge(reduction=tf.keras.losses.Reduction.NONE))

@tf.function
def fuzzy_loss(y_true, y_pred, loss_function):
    i = tf.constant(0)
    acc_outer = tf.constant(0.0)
    
    def outer_loop(i, acc_outer):
        y, _ = tf.unique(y_true[i])
        cuts = tf.sort(y, direction="DESCENDING")
        
        n_cuts = cuts.shape[0]
        j = tf.constant(0)
        acc_cut = tf.constant(0.0)
        
        def cut_loop(j, acc_cut):
            indices = tf.where(y_true[i] >= cuts[j])
            cut_size = indices.shape[0]
            
            k = tf.constant(0)
            acc_inner = tf.constant(0.0)
            
            def inner_loop(k, acc_inner):
                val = tf.reshape(tf.tile(y_pred[i,:], [cut_size]), (cut_size, y_true.shape[1]))
                loss = tf.reduce_min(loss_function(indices, val))
                return (tf.add(k,1), tf.add(acc_inner,loss))
            
            c_inner = lambda k, _: tf.less(k, cut_size)
            b_inner = lambda k, acc_inner: inner_loop(k, acc_inner)
            _, acc_inner = tf.while_loop(c_inner, b_inner, [k, acc_inner])
            return (tf.add(j, 1), tf.divide(tf.multiply(cuts[j] - cuts[j+1], acc_inner), cut_size))
            
        
        c_cut = lambda j, _ : tf.less(j, n_cuts-1)
        b_cut = lambda j, acc_cut: cut_loop(j, acc_cut)
        _, acc_cut = tf.while_loop(c_cut, b_cut, [j,acc_cut])
        
        return (tf.add(i,1), tf.add(acc_outer, acc_cut))
    
    c_outer = lambda i, _ : tf.less(i, y_pred.shape[0])
    b_outer = lambda i, acc_outer: outer_loop(i, acc_outer)
    _, acc_outer = tf.while_loop(c_outer, b_outer, [i, acc_outer])
    return acc_outer