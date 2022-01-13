import sys
import os
from pathlib import Path
CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(str(PATH.parent.parent.parent.absolute()))
sys.path.append(str(PATH.parent.parent.parent.parent.absolute()))
sys.path.append(CURR_DIR)

import argparse
import json
from pprint import pprint
from collections import Counter
import torch

import numpy as np
import random
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from URE_mnli.relation_classification.mnli import NLIRelationClassifierWithMappingHead, REInputFeatures
from URE_mnli.relation_classification.tacred import *
from URE_mnli.relation_classification import arguments
from URE_mnli.relation_classification.utils import find_optimal_threshold, apply_threshold,load,save,set_global_random_seed

CLASSIFIERS = {"mnli-mapping": NLIRelationClassifierWithMappingHead}
set_global_random_seed(arguments.seed)  # 设置随机种子


def top_k_accuracy(output, labels, k=5, id2labels=None):
    """
    top_k_accuracy for whole output
    若id2labels非空, 则可返回topk的predict出来的relation
    """
    preds = np.argsort(output)[:, ::-1][:, :k]
    total = len(preds)
    right = 0
    predict_relations = []
    for l, p in zip(labels, preds):
        if id2labels is not None:
            predict_relations.append(id2labels[p[-1]])
        if l in p:
            right +=1
    if id2labels is not None:
        return right/total, predict_relations
    return right/total

    # return sum(l in p and l > 0 for l, p in zip(labels, preds)) / (labels > 0).sum()


parser = argparse.ArgumentParser(
    prog="run_evaluation",
    description="Run a evaluation for each configuration.",
)
parser.add_argument(
    "--input_file",
    type=str,
    default=arguments.run_evaluation_path,
    help="Dataset file.",
)
parser.add_argument(
    "--config",
    type=str,
    default=arguments.config_path,
    help="Configuration file for the experiment.",
)
parser.add_argument("--basic", action="store_true", default=False)

args = parser.parse_args()
unique_info = "Config="+arguments.config_path.split("/")[-1].split(".")[0]+"Time="+arguments.current_time
args.unique_info = unique_info
labels2id = (
    {label: i for i, label in enumerate(TACRED_LABELS)}
    if not args.basic
    else {label: i for i, label in enumerate(TACRED_BASIC_LABELS)}
)

# id2labels
id2labels = dict(zip(
    list(labels2id.values()),
    list(labels2id.keys())
))

with open(arguments.split_path) as file:
    split_ids = file.readlines()
    split_ids = [item.replace("\n","") for item in split_ids]

with open(args.input_file, "rt") as f:
    features, labels, relations = [], [],[]
    for line in json.load(f):
        id = line['id']
        if arguments.split:
            if id not in split_ids:
                continue
        line["relation"] = (
            line["relation"] if not args.basic else TACRED_BASIC_LABELS_MAPPING.get(line["relation"], line["relation"])
        )
        features.append(
            REInputFeatures(
                subj=" ".join(line["token"][line["subj_start"] : line["subj_end"] + 1])
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                obj=" ".join(line["token"][line["obj_start"] : line["obj_end"] + 1])
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                pair_type=f"{line['subj_type']}:{line['obj_type']}",
                context=" ".join(line["token"])
                .replace("-LRB-", "(")
                .replace("-RRB-", ")")
                .replace("-LSB-", "[")
                .replace("-RSB-", "]"),
                label=line["relation"],
            )
        )
        relations.append(line["relation"])
        labels.append(labels2id[line["relation"]])

# dict_keys(['text', 'rel', 'subj', 'obj', 'subj_type', 'obj_type', 'top1', 'top2', 'label', 'pos_or_not'])


if arguments.save_dataset_name is not None:
    dataset = {
    'text':[],
    'rel':[],
    'subj':[],
    'obj':[],
    'subj_type':[],
    'obj_type':[],
    }
    assert len(features)==len(relations)
    for feat,rel in zip(features,relations):
        dataset['text'].append(feat.context)
        dataset['rel'].append(rel)
        dataset['subj'].append(feat.subj)
        dataset['obj'].append(feat.obj)
        subj_type,obj_type = feat.pair_type.split(":")
        dataset['subj_type'].append(subj_type)
        dataset['obj_type'].append(obj_type)
    save(dataset,os.path.join("/home/tywang/myURE/URE/TACRED/tac_six_key",arguments.save_dataset_name))


labels = np.array(labels)  # feature的label


# 根据select_ratio随机选择数据
if arguments.selected_ratio is not None:
    L = len(features)
    indexes = [i for i in range(L)]
    random.shuffle(indexes)
    indexes = indexes[:int(L*arguments.selected_ratio)]
    features = [features[i] for i in indexes]
    labels = labels[np.array(indexes)]


with open(args.config, "rt") as f:
    config = json.load(f)

LABEL_LIST = TACRED_BASIC_LABELS if args.basic else TACRED_LABELS

for configuration in config:
    n_labels = len(LABEL_LIST)
    os.makedirs(f"experiments/{configuration['name']}", exist_ok=True)
    _ = configuration.pop("negative_threshold", None)
    classifier = CLASSIFIERS[configuration["classification_model"]](negative_threshold=0.0, **configuration)
    output = classifier(
        features,
        batch_size=configuration["batch_size"],
        multiclass=configuration["multiclass"],
    )
    if not "use_threshold" in configuration or configuration["use_threshold"]:
        if arguments.get_optimal_threshold:
            optimal_threshold, _ = find_optimal_threshold(labels, output)  
            # 0.01 dev optimal_threshold = 0.922922 
            # selected by oscar 0.01 dev optimal_threshold = 0.96096
        else:
            optimal_threshold = 0.922922 # set default threshold
        output_,applied_threshold_output = apply_threshold(output, threshold=optimal_threshold)
    else:
        output_ = output.argmax(-1)
    pre, rec, f1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        labels, output_, average="micro", labels=list(range(1, n_labels))
    )

    configuration["precision"] = pre
    configuration["recall"] = rec
    configuration["f1-score"] = f1

    configuration["top-1"], top1_p_rel = top_k_accuracy(applied_threshold_output, labels, k=1, id2labels=id2labels)
    configuration["top-2"], top2_p_rel = top_k_accuracy(applied_threshold_output, labels, k=2, id2labels=id2labels)
    configuration["top-3"], top3_p_rel = top_k_accuracy(applied_threshold_output, labels, k=3, id2labels=id2labels)
    for i in range(1,4):
        print("top{} acc={:.4f}".format(i, configuration["top-{}".format(i)]))


    del classifier
    torch.cuda.empty_cache()
