import tensorflow as tf


## source https://stackoverflow.com/questions/62793043/tensorflow-implementation-of-nt-xent-contrastive-loss-function

# Define the contrastive loss function, NT_Xent (Tensorflow version)
def NT_Xent_tf(zi, zj, tau=1):
    """ Calculates the contrastive loss of the input data using NT_Xent. The
    equation can be found in the paper: https://arxiv.org/pdf/2002.05709.pdf
    (This is the Tensorflow implementation of the standard numpy version found
    in the NT_Xent function).

    Args:
        zi: One half of the input data, shape = (batch_size, feature_1, feature_2, ..., feature_N)
        zj: Other half of the input data, must have the same shape as zi
        tau: Temperature parameter (a constant), default = 1.

    Returns:
        loss: The complete NT_Xent constrastive loss
    """
    z = tf.cast(tf.concat((zi, zj), 0), dtype=tf.float32)
    loss = 0
    for k in range(zi.shape[0]):
        # Numerator (compare i,j & j,i)
        i = k
        j = k + zi.shape[0]
        # Instantiate the cosine similarity loss function
        cosine_sim = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        sim = tf.squeeze(- cosine_sim(tf.reshape(z[i], (1, -1)), tf.reshape(z[j], (1, -1))))
        numerator = tf.math.exp(sim / tau)

        # Denominator (compare i & j to all samples apart from themselves)
        sim_ik = - cosine_sim(tf.reshape(z[i], (1, -1)), z[tf.range(z.shape[0]) != i])
        sim_jk = - cosine_sim(tf.reshape(z[j], (1, -1)), z[tf.range(z.shape[0]) != j])
        denominator_ik = tf.reduce_sum(tf.math.exp(sim_ik / tau))
        denominator_jk = tf.reduce_sum(tf.math.exp(sim_jk / tau))

        # Calculate individual and combined losses
        loss_ij = - tf.math.log(numerator / denominator_ik)
        loss_ji = - tf.math.log(numerator / denominator_jk)
        loss += loss_ij + loss_ji

    # Divide by the total number of samples
    loss /= z.shape[0]

    return loss
