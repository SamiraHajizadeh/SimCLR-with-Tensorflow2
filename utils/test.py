import numpy as np
from utils.evaluation_metrics import get_top_k_accuracy

def test(model, test_dataset, k=1, print_results=False, return_all_accs=False):
    acc = []
    for X, labels in test_dataset:
        acc.append(get_top_k_accuracy(X, labels, model, k, print_acc=False))
    mean_acc = np.mean(acc)
    if print_results:
        print(f'Average accuracy of model over the test dataset {mean_acc}')
    return acc if return_all_accs else mean_acc
