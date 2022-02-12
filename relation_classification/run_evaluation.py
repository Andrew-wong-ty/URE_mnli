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
from sklearn.metrics import precision_recall_fscore_support
from URE.clean_data.clean import get_format_train_text
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
    features, labels, relations,subj_pos,obj_pos = [], [],[],[],[]
    for line in json.load(f):
        id = line['id']
        if arguments.split and arguments.selected_ratio is None:
            if id not in split_ids:
                continue
        line["relation"] = (
            line["relation"] if not args.basic else TACRED_BASIC_LABELS_MAPPING.get(line["relation"], line["relation"])
        )
        subj_posistion= [line["subj_start"] , line["subj_end"]]
        subj_pos.append(subj_posistion)
        obj_posistion= [line["obj_start"] , line["obj_end"]]
        obj_pos.append(obj_posistion)
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




labels = np.array(labels)  # feature的label
print(Counter(relations))


# 根据select_ratio随机选择数据
if arguments.selected_ratio is not None:
    L = len(features)
    indexes = [i for i in range(L)]
    random.shuffle(indexes)
    indexes = indexes[:int(L*arguments.selected_ratio)]
    features = [features[i] for i in indexes]
    relations = [relations[i] for i in indexes]
    labels = labels[np.array(indexes)]


with open(args.config, "rt") as f:
    config = json.load(f)

LABEL_LIST = TACRED_BASIC_LABELS if args.basic else TACRED_LABELS

for configuration in config:
    n_labels = len(LABEL_LIST)
    os.makedirs(f"experiments/{configuration['name']}", exist_ok=True)
    _ = configuration.pop("negative_threshold", None)
    classifier = CLASSIFIERS[configuration["classification_model"]](negative_threshold=0.0, **configuration)
    output,template_socre,template_sorted, template2label = classifier(
        features,
        batch_size=configuration["batch_size"],
        multiclass=configuration["multiclass"],
    )
    #save(template2label,"/home/tywang/myURE/URE/O2U_bert/tac_data/whole/train_template2label.pkl")
    if not "use_threshold" in configuration or configuration["use_threshold"]:
        if arguments.get_optimal_threshold:
            optimal_threshold, _ = find_optimal_threshold(labels, output)  
            print("optimal threshold:",optimal_threshold)
            # 0.01 dev optimal_threshold = 0.9379379379379379(13)  (没有finetune) 
            # 0.01 dev 0.01trai一半pos一半neg finetune  1.0
            # 0.01 dev 0.01train全pos finetune  0.997997997997998
            # 0.01 dev 0.01train全neg finetune  0.8188188188188188
            # selected by oscar 0.01 dev optimal_threshold = 0.96096
            # re-tac 0.01dev optimal_threshold 0.8758758758758759
        else:
            
            optimal_threshold = arguments.default_optimal_threshold # set default threshold
            print("use threshold:{}".format(optimal_threshold))
        top1,applied_threshold_output = apply_threshold(output, threshold=optimal_threshold)
    else:
        top1 = output.argmax(-1)

    pre, rec, f1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        labels, top1, average="micro", labels=list(range(1, n_labels))
    )
    top1_acc = sum(top1==labels)/len(labels)
    top1_p_rel = [id2labels[item] for item in top1]


    configuration["precision"] = pre
    configuration["recall"] = rec
    configuration["f1-score"] = f1

    configuration["top-1"] = top1_acc
    configuration["top-2"], top2_p_rel = top_k_accuracy(applied_threshold_output, labels, k=2, id2labels=id2labels)
    configuration["top-3"], top3_p_rel = top_k_accuracy(applied_threshold_output, labels, k=3, id2labels=id2labels)
    print("labeled f1:{:.4f}".format(f1))
    print("precision:{:.4f}".format(pre))
    print("recall:{:.4f}".format(rec))
    for i in range(1,4):
        print("top{} acc={:.4f}".format(i, configuration["top-{}".format(i)]))
    

    

    if arguments.generate_data is not None:
        label2id = load(arguments.label2id_path)
        id2label = dict(zip(label2id.values(),label2id.keys()))
        # save(id2label,"/home/tywang/myURE/URE_mnli/temp_files/analysis_0.01510/id2label.pkl")
        dataset = {
        'text':[],
        'rel':[],
        'subj':[],
        'obj':[],
        'subj_pos':subj_pos,
        'obj_pos':obj_pos,
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
        for text,subj, subj_p, obj,obj_p in zip(dataset['text'],dataset['subj'],dataset['subj_pos'],dataset['obj'],dataset['obj_pos']):
            assert ' '.join(text.split()[subj_p[0]:subj_p[1]+1])==subj
            assert ' '.join(text.split()[obj_p[0]:obj_p[1]+1])==obj
        # save(dataset,"/home/tywang/myURE/URE/clean_data/test_data/tac_clean_testdata.pkl")
        # dataset, etags = get_format_train_text(dataset,True)
        # save(etags,"/home/tywang/myURE/URE/O2U_bert/tac_data/train_tags.pkl")
        # tags = load("/home/tywang/myURE/URE/O2U_bert/tac_data/train_tags.pkl")
        dataset['template'] = template_sorted
        dataset['index'] = [i for i in range(len(dataset['text']))]
        dataset['label'] = [label2id[item] for item in relations]
        dataset['top1'] = [label2id[item] for item in top1_p_rel]
        dataset['top2'] = [label2id[item] for item in top2_p_rel]
        dataset['top3'] = [label2id[item] for item in top3_p_rel]
        top1_acc = sum(np.array(dataset['label'])==np.array(dataset['top1']))/len(dataset['label'])
        _, _, f1_, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
        dataset['label'], dataset['top1'] , average="micro", labels=list(range(0,41))
        )
        print("top1 acc: ",top1_acc)
        print("labeled f1: ",f1_)
        # save(dataset,"/home/tywang/myURE/URE_mnli/relation_classification/研究数据/n0.1train.pkl")
        #save(dataset,"/home/tywang/myURE/URE_mnli/temp_files/analysis_0.01510/tac_{}_num{}_top1_{:.4f}.pkl".format(arguments.mode,len(dataset['text']),top1_acc))
        
    # del classifier
    # torch.cuda.empty_cache()
