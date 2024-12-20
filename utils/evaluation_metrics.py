import tensorflow as tf
import numpy as np

def get_top_k_accuracy(X, y, model, k=5, print_acc=True):
    y_true = tf.cast(y, dtype=tf.int32)
    y_pred = tf.keras.layers.Softmax()(model(X))
    out = tf.nn.in_top_k(y_true, y_pred, k=k)
    top_k_accuracy = np.sum(out) / out.shape[0]
    if print_acc:
        print(f"Top {k} Accuracy: {top_k_accuracy}")
    return top_k_accuracy
