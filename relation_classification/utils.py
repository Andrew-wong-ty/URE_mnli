import numpy as np
import pickle
import json
import copy
from sklearn.metrics import f1_score, precision_recall_fscore_support
import random
import torch



def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def save(obj, path_name):
    with open(path_name, 'wb') as file:
        pickle.dump(obj, file)
def load_json(path):
    with open(path, "rt") as f:
        data = json.load(f)
    return data

def load(path_name: object) -> object:
    with open(path_name, 'rb') as file:
        return pickle.load(file)

def f1_score_(labels, preds, n_labels=42):
    return f1_score(labels, preds, labels=list(range(1, n_labels)), average="micro")


def precision_recall_fscore_(labels, preds, n_labels=42):
    p, r, f, _ = precision_recall_fscore_support(labels, preds, labels=list(range(1, n_labels)), average="micro")
    return p, r, f


def find_uppercase(str1:list, str2:list):
	"""
		str1 is a longer text
		str2(only lower case) is text included in str1
		find text in str1 that consist with str2, ignoring case. Then return the text in str1
	"""
	l2 = len(str2)
	for i in range(len(str1[:-l2])+1):
		if ' '.join(str1[i:i+l2]).lower().split()==str2:
			return ' '.join(str1[i:i+l2])
	return ' '.join(str2)

def apply_threshold(output, threshold=0.0, ignore_negative_prediction=True):
    """Applies a threshold to determine whether is a relation or not"""
    output_ = output.copy()
    if ignore_negative_prediction:
        output_[:, 0] = 0.0
    activations = (output_ >= threshold).sum(-1).astype(np.int)  # 如果没有一个pos rel的 prob>threshold  , 那么归为no-rel
    output_[activations == 0, 0] = 1.00

    
    applied_threshold_output = copy.deepcopy(output_)

    return output_.argmax(-1),applied_threshold_output  # matrix


def find_optimal_threshold(labels, output, granularity=1000, metric=f1_score_):
    thresholds = np.linspace(0, 1, granularity)
    values = []
    for t in thresholds:
        preds,_ = apply_threshold(output, threshold=t)
        values.append(metric(labels, preds))

    best_metric_id = np.argmax(values)
    best_threshold = thresholds[best_metric_id]

    return best_threshold, values[best_metric_id]
