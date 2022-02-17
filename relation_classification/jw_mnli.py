from pickle import load
import sys
import os
from pathlib import Path
CURR_FILE_PATH = (os.path.abspath(__file__))
PATH = Path(CURR_FILE_PATH)
CURR_DIR = str(PATH.parent.absolute())
sys.path.append(str(PATH.parent.parent.parent.absolute()))
sys.path.append(str(PATH.parent.parent.parent.parent.absolute()))
sys.path.append(str(PATH.parent.parent.parent.parent.parent.absolute()))
sys.path.append(CURR_DIR)

# print(sys.path)
# from a2t.slot_classification.utils.we_scorer.utils import find_arg_span
# from numpy.random.mtrand import random
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from collections import defaultdict
from genericpath import exists
from pprint import pprint
from typing import Dict, List
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    T5ForConditionalGeneration,
)

from URE_mnli.base import Classifier, np_softmax

import argparse
import pickle
import math
import random

# from tacred import TACREDClassifier

import os

def save(obj,path_name):
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def load(path_name: object) -> object:
    with open(path_name,'rb') as file:
        return pickle.load(file)


@dataclass
class REInputFeatures:
    index:int
    subj: str
    obj: str
    context: str
    pair_type: str = None
    label: str = None

# ATTENTION
config={
    'data_path':'/data/jwwang/output/pretrain-wiki-type-extend-v2/test_premnil.pkl',# 输入的数据
    'save_dir': '/home/tywang/myURE/URE_mnli/temp_files/jw_test/', #输出的路径
    'model_path':"/data/transformers/microsoft_deberta-v2-xlarge-mnli"  # "/data/transformers/microsoft_deberta-v2-xlarge-mnli" #model路径
}

class _NLIRelationClassifier(Classifier):
    def __init__(
        self,
        labels: List[str],
        *args,
        pretrained_model: str = "roberta-large-mnli",
        use_cuda=True,
        half=False,
        verbose=True,
        negative_threshold=0.95,
        negative_idx=0,
        max_activations=np.inf,
        valid_conditions=None,
        **kwargs,
    ):
        super().__init__(
            labels,
            pretrained_model=pretrained_model,
            use_cuda=use_cuda,
            verbose=verbose,
            half=half,
        )

        self.negative_threshold = negative_threshold
        self.negative_idx = negative_idx
        self.max_activations = max_activations
        self.n_rel = len(labels)


        def idx2label(idx):
            return self.labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def _verify_thredhold(self, thred):
        self.negative_threshold=thred

    def _initialize(self, pretrained_model):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

        # print(self.device)
        # device = torch.device("cuda:2" if torch.cuda.is_available()  else "cpu")

        self.model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

        # print(next(self.model.parameters()).device) 

        self.config = AutoConfig.from_pretrained(pretrained_model)
        self.ent_pos = self.config.label2id.get("ENTAILMENT", self.config.label2id.get("entailment", None))

        # print(self.config.label2id)

        if self.ent_pos is None:
            raise ValueError("The model config must contain ENTAILMENT label in the label2id dict.")
        else:
            self.ent_pos = int(self.ent_pos)

    def _run_batch(self, batch, multiclass=False):
        # model run
        # device = torch.device("cuda:2" if torch.cuda.is_available()  else "cpu")
        with torch.no_grad():

            # print(batch)
            input_ids = self.tokenizer.batch_encode_plus(batch, padding=True, truncation=True)


            input_ids = torch.tensor(input_ids["input_ids"]).to(self.device)

            # print(input_ids.shape[0])

            # print(next(self.model.parameters()).device) 
            # print(input_ids.device)

            output = self.model(input_ids)[0].detach().cpu().numpy()
            if multiclass:
                output = np.exp(output) / np.exp(output).sum(
                    -1, keepdims=True
                )  # np.exp(output[..., [self.cont_pos, self.ent_pos]]).sum(-1, keepdims=True)
            
            # print(self.labels)
            # print(output.shape)=> 68,3
            output = output[..., self.ent_pos].reshape(-1)
            # print(output.shape)=>1,68
        return output

    def __call__(
        self,
        features: List[REInputFeatures],
        batch_size: int = 1,
        multiclass=False,
    ):
        #call0
        if not isinstance(features, list):
            features = [features]

        batch, outputs = [], []
        sentences_ids=[]

        # cnt=0
        for i, feature in tqdm(enumerate(features), total=len(features)):
            # print(next(self.model.parameters()).device) 
            sentences = []
            sentence_id=[]
            for label_template in self.labels:
                # if self._apply_valid2(feature,label_template):
                sentences.append(f"{feature.context} {self.tokenizer.sep_token} {label_template.format(subj=feature.subj, obj=feature.obj)}.")
                sentence_id.append(self.new_labels2id[label_template])

            # print(sentences)

            batch.extend(sentences)
            sentences_ids.append(sentence_id)
            # cnt+=1

            if (i + 1) % batch_size == 0:
                output = self._run_batch(batch, multiclass=multiclass)            
                output1=np.zeros([len(sentences_ids),92])
                cnt=0
                for index,i in enumerate(sentences_ids):
                    for index2,i2 in enumerate(i):
                        output1[index, i2]=output[cnt]
                        cnt+=1
                outputs.append(output1)
                batch = []
                sentences_ids=[]

            

        if len(batch) > 0:
            output = self._run_batch(batch, multiclass=multiclass)            
            output1=np.zeros([len(sentences_ids),92])
            cnt=0
            for index,i in enumerate(sentences_ids):
                for index2,i2 in enumerate(i):
                    output1[index, i2]=output[cnt]
                    cnt+=1
            outputs.append(output1) 
            batch = []
            sentences_ids=[]

        outputs = np.vstack(outputs)

        return outputs
    def _apply_negative_threshold(self, probs):
        activations = (probs >= self.negative_threshold).sum(-1).astype(int)
        idx = np.logical_or(
            activations == 0, activations >= self.max_activations
        )  # If there are no activations then is a negative example, if there are too many, then is a noisy example
        probs[idx, self.negative_idx] = 1.00
        return probs

    def _apply_valid_conditions(self, probs, features: List[REInputFeatures]):
        mask_matrix = np.stack(
            [self.valid_conditions.get(feature.pair_type, np.zeros(self.n_rel)) for feature in features],
            axis=0,
        )
        probs = probs * mask_matrix

        return probs
    
    def _apply_valid2(self, features: REInputFeatures,label_template):
        # TODO apply valid twice
        type_pairs=[]
        rels=self.template_mapping_reverse[label_template]
        if rels[0] not in self.rules:
            return True
        for rel in rels:
            # if rel=='no_relation':
            #     return True
            type_pairs.extend(self.rules[rel])

        # print(features.pair_type)
        # print(type_pairs)

        if features.pair_type in type_pairs:
            return True
        else:
            return False


    def predict(
        self,
        contexts: List[str],
        batch_size: int = 1,
        return_labels: bool = True,
        return_confidences: bool = False,
        topk: int = 1,
    ):
        output = self(contexts, batch_size)
        topics = np.argsort(output, -1)[:, ::-1][:, :topk] #argsort和sort 获取index和获取item的区别
        if return_labels:
            topics = self.idx2label(topics)
        if return_confidences:
            topics = np.stack((topics, np.sort(output, -1)[:, ::-1][:, :topk]), -1).tolist()
            topics = [
                [(int(label), float(conf)) if not return_labels else (label, float(conf)) for label, conf in row]
                for row in topics
            ]
        else:
            topics = topics.tolist()
        if topk == 1:
            topics = [row[0] for row in topics]

        return topics


class NLIRelationClassifierWithMappingHead(_NLIRelationClassifier):
    def __init__(
        self,
        labels: List[str],
        template_mapping: Dict[str, str],
        pretrained_model: str = "roberta-large-mnli",
        valid_conditions: Dict[str, list] = None,
        *args,
        **kwargs,
    ):


        self.template_mapping_reverse = defaultdict(list) #template2rel
        for key, value in template_mapping.items():
            for v in value:
                self.template_mapping_reverse[v].append(key)
        self.new_topics = list(self.template_mapping_reverse.keys())

        self.target_labels = labels #rel
        self.new_labels2id = {t: i for i, t in enumerate(self.new_topics)} # template2id
        self.mapping = defaultdict(list) #rel2templateid
        for key, value in template_mapping.items():
            self.mapping[key].extend([self.new_labels2id[v] for v in value])

        super().__init__(
            self.new_topics,
            *args,
            pretrained_model=pretrained_model,
            valid_conditions=None,
            **kwargs,
        )

        self.rules=valid_conditions

        if valid_conditions:
            self.valid_conditions = {} # condition2relid
            self.rel2id = {r: i for i, r in enumerate(labels)}
            self.n_rel = len(self.rel2id)
            for i in ruleofnull:
                    self.valid_conditions[i]=np.ones(self.n_rel)
            for relation, conditions in valid_conditions.items():

                if relation not in self.rel2id:
                    continue
                for condition in conditions:
                    if condition not in self.valid_conditions:
                        self.valid_conditions[condition] = np.zeros(self.n_rel)
                        # self.valid_conditions[condition][self.rel2id["no_relation"]] = 1.0
                        # if condition[0:6]=="PERSON":
                        #     for i in rulesofperson:
                        #         self.valid_conditions[condition][self.rel2id[i]] = 1.0
                        # else:
                        #     for i in rulesoforg:
                        #         self.valid_conditions[condition][self.rel2id[i]] = 1.0
                        # for i in relofnull:
                        #     self.valid_conditions[condition][self.rel2id[i]] = 1.0
                    self.valid_conditions[condition][self.rel2id[relation]] = 1.0

        else:
            self.valid_conditions = None

        def idx2label(idx):
            return self.target_labels[idx]

        self.idx2label = np.vectorize(idx2label)

    def __call__(self, features: List[REInputFeatures], batch_size=1, multiclass=True):
        # call1
        # print(self.target_labels)
        # print(self.mapping)
        outputs = super().__call__(features, batch_size, multiclass)
        outputs = np.hstack(
            [
                np.max(outputs[:, self.mapping[label]], axis=-1, keepdims=True) #mapping rel2templateid
                if label in self.mapping
                else np.zeros((outputs.shape[0], 1))
                for label in self.target_labels
            ]
        )
        outputs = np_softmax(outputs) if not multiclass else outputs #归一化 softmax

        # outputs.shape=>(1, 42)

        # TODO apply valid twice
        if self.valid_conditions:
            outputs = self._apply_valid_conditions(outputs, features)

        outputs = self._apply_negative_threshold(outputs)

        # outputs=>bs * poss
        return outputs



ruleofnull=['all']

def TE():
    data_path=config['data_path']
    sents_batch_size=10000

    data=load(data_path)
    sentences=data['text']
    subjs=data['subj']
    objs=data['obj']
    subjs_type=data['subj_type']
    objs_type=data['obj_type']
    rels=data['rel']
    index=data['index']



    labels = ['place served by transport hub', 'mountain range', 'religion', 'participating team', \
     'contains administrative territorial entity', 'head of government', 'country of citizenship', \
     'original network', 'heritage designation', 'performer', 'participant of', 'position held', \
     'has part', 'location of formation', 'located on terrain feature', 'architect', \
     'country of origin', 'publisher', 'director', 'father', 'developer', 'military branch', \
     'mouth of the watercourse', 'nominated for', 'movement', 'successful candidate', 'followed by',\
     'manufacturer', 'instance of', 'after a work by', 'member of political party', \
     'licensed to broadcast to', 'headquarters location', 'sibling', 'instrument', 'country', \
     'occupation', 'residence', 'work location', 'subsidiary', 'participant', 'operator', \
     'characters', 'occupant', 'genre', 'operating system', 'owned by', 'platform', 'tributary', \
     'winner', 'said to be the same as', 'composer', 'league', 'record label', 'distributor', \
     'screenwriter', 'sports season of league or competition', 'taxon rank', 'location', \
     'field of work', 'language of work or name', 'applies to jurisdiction', 'notable work', \
     'located in the administrative territorial entity', 'crosses', \
     'original language of film or TV show', 'competition class', 'part of', 'sport', \
     'constellation', 'position played on team / speciality', 'located in or next to body of water', \
     'voice type', 'follows', 'spouse', 'military rank', 'mother', 'member of', 'child', \
     'main subject']

     

    label2id={}
    for i,j in enumerate(labels):
        label2id[j]=i

    template_mapping = {'place served by transport hub': ['{subj} is the place that served by a transport hub in {obj}.'], 
    'mountain range': ['{subj} mountain range is in the {obj}.', 
    '{subj} mountain range is on the {obj}.', '{subj} mountain range is part of the {obj}.'], 
    'religion': ["{obj} is {subj}'s religion."], 
    'participating team': ['{obj} team participated in {subj}.', '{obj} rival participated in {subj}.'], 
    'contains administrative territorial entity': ['{obj} place is the terrioty of {subj}.'], 
    'head of government': ['{obj} is the government head of {subj}.'], 
    'country of citizenship': ['{obj} country does {subj} has a citizenship of.'], 
    'original network': ['{obj} is the original network of {subj}.'], 
    'heritage designation': ['{subj} heritage designation is listed on the {obj}.'], 
    'performer': ['{obj} are performers of " {subj} ".'], 
    'participant of': ['{subj} participated in {obj}.', '{obj} event did {subj} participate in.'], 
    'position held': ['{obj} position is held by {subj}.'], 
    'has part': ['{subj} does {obj} belong to.'], 
    'location of formation': ['{obj} is {subj} formed.'], 
    'located on terrain feature': ['{obj} is the terrain feature {subj} located in.'], 
    'architect': ['{obj} is the architect of {subj}.'], 
    'country of origin': ["{obj} is {subj}'s country of origin."], 
    'publisher': ['{obj} is the publisher of " {subj} ".'], 'director': ['{obj} is the director of " {subj} ".'], 
    'father': ["{obj} is {subj}'s father."], 'developer': ['{obj} is the developer of " {subj} ".'], 
    'military branch': ['{obj} military branch does {subj} work for.'], 
    'mouth of the watercourse': ['{subj} is the mouth of the watercourse {obj}.'], 
    'nominated for': ['{obj} are " {subj} " nominated for.', '{subj} is the nominee of {obj}.'], 
    'movement': ['{obj} is movement of {subj}.'], 'successful candidate': ['{obj} is the successful candidate of {subj}.'], 
    'followed by': ['{subj} is before " {obj} ".', '{subj} is followed by " {obj} ".'], 
    'manufacturer': ['{obj} is the manufacturer of {subj}.'], 
    'instance of': ['{subj} is an instance of {obj}.', '{obj} is the {subj}.'], 
    'after a work by': ['{subj} is created by " {obj} ".', '{subj} is based on " {obj} ".'], 
    'member of political party': ['{obj} political party does {subj} belong to.'], 
    'licensed to broadcast to': ['{subj} is licensed to {obj}.'], 'headquarters location': ['{obj} is the headquarter of {subj}.'], 
    'sibling': ["{obj} are {subj}'siblings.", "{subj} are {obj}'s siblings."], 
    'instrument': ['{obj} instruments does {subj} play.'], 'country': ['{obj} country does {subj} belong to.'], 
    'occupation': ["{obj} is {subj}'s occupation."], 'residence': ['{obj} does {subj} live in.'], 
    'work location': ['{obj} does {subj} work in.'], 'subsidiary': ['{obj} organization is the subsidiary of {subj}.'], 
    'participant': ['{obj} are participants of {subj}.'], 'operator': ['{obj} are operators of {subj}.'], 
    'characters': ['{obj} are the characters of {subj}.'], 'occupant': ['{obj} teams are occupants of {subj}.'], 
    'genre': ['{obj} is the genre of " {subj} ".'], 'operating system': ['{obj} are operating systems of {subj}.'], 
    'owned by': ['{obj} own {subj}.'], 'platform': ['{subj} are platforms of {obj}.'], 
    'tributary': ['{obj} are tributaries of {subj}.'], 'winner': ['{obj} are the winners of {subj}.'], 
    'said to be the same as': ['{obj} are said to be the same as {subj}.'], 'composer': ['{obj} are composers of {subj}.'], 
    'league': ['{obj} is the league of {subj}.'], 'record label': ['{obj} is the record label of {subj}.'], 
    'distributor': ['{obj} are distributors of {subj}.'], 'screenwriter': ['{obj} are screenwriters of {subj}.'], 
    'sports season of league or competition': ['{subj} seasons of {obj} are mentioned.'], 
    'taxon rank': ['{obj} is taxon rank of {subj}.'], 'location': ['{obj} did {subj} held.'], 
    'field of work': ["{obj} are {subj}'s fields of work."], 
    'language of work or name': ['{obj} is the language of the work " {subj} ".', '{obj} is the language of the name " {subj} ".'], 
    'applies to jurisdiction': ['{obj} is the jurisdiction of {subj} applied to.'], 
    'notable work': ['{obj} are notable works of {subj}.'], 
    'located in the administrative territorial entity': ['{obj} is the administrative territorial entity {subj} located in.'], 
    'crosses': ['{subj} cross {obj}.'], 
    'original language of film or TV show': ['{obj} is the original language of the film " {subj} ".', 
    '{obj} is the original language of the TV show " {subj} ".'], 
    'competition class': ['{obj} is the competition class of {subj}.'], 
    'part of': ['{subj} is a part of {obj}.'], 'sport': ['{obj} sports does {subj} play.'], 
    'constellation': ['{subj} are in the constellation of " {obj} ".'], 
    'position played on team / speciality': ['{obj} position does {subj} play on the team.'], 
    'located in or next to body of water': ['{obj} body of water is {subj} located in.'], 
    'voice type': ['{obj} is the voice type of {subj}.'], 'follows': ['{subj} is after " {obj} ".', 
    '{subj} follows " {obj} ".'], 'spouse': ["{obj} is {subj}'s spouse."], 
    'military rank': ['{obj} is the military rank of {subj}.'], 'mother': ["{obj} is {subj}'s mother."], 
    'member of': ['{subj} is a member of {obj}.'], 'child': ["{obj} are {subj}'s children."], 
    'main subject': ['{obj} is the main subject of " {subj} ".']}

    rules={'place served by transport hub': ['FAC:GPE'], 'mountain range': ['LOC:LOC'], 'religion': ['PERSON:NORP'], 'participating team': ['EVENT:ORG'], 'contains administrative territorial entity': ['GPE:GPE'], 'head of government': ['GPE:PERSON'], 'country of citizenship': ['PERSON:GPE'], 'original network': ['WORK_OF_ART:ORG'], 'heritage designation': ['FAC:ORG'], 'performer': ['WORK_OF_ART:PERSON'], 'participant of': ['PERSON:EVENT'], 'position held': ['PERSON:ORG'], 'has part': ['ORG:PERSON'], 'location of formation': ['ORG:GPE'], 'located on terrain feature': ['FAC:LOC'], 'architect': ['FAC:PERSON'], 'country of origin': ['WORK_OF_ART:GPE'], 'publisher': ['WORK_OF_ART:ORG'], 'director': ['WORK_OF_ART:PERSON'], 'father': ['PERSON:PERSON'], 'developer': ['WORK_OF_ART:ORG'], 'military branch': ['PERSON:ORG'], 'mouth of the watercourse': ['LOC:LOC'], 'nominated for': ['WORK_OF_ART:WORK_OF_ART'], 'movement': ['PERSON:NORP'], 'successful candidate': ['SocietalEvent:PERSON'], 'followed by': ['WORK_OF_ART:WORK_OF_ART'], 'manufacturer': ['PRODUCT:ORG'], 'instance of': ['WORK_OF_ART:WORK_OF_ART'], 'after a work by': ['WORK_OF_ART:PERSON'], 'member of political party': ['PERSON:ORG'], 'licensed to broadcast to': ['ORG:GPE'], 'headquarters location': ['ORG:GPE'], 'sibling': ['PERSON:PERSON'], 'instrument': ['PERSON:WORK_OF_ART'], 'country': ['GPE:GPE'], 'occupation': ['PERSON:PERSON'], 'residence': ['PERSON:GPE'], 'work location': ['PERSON:GPE'], 'subsidiary': ['ORG:ORG'], 'participant': ['EVENT:PERSON'], 'operator': ['FAC:ORG'], 'characters': ['WORK_OF_ART:PERSON'], 'occupant': ['FAC:ORG'], 'genre': ['ORG:ORG'], 'operating system': ['PRODUCT:PRODUCT'], 'owned by': ['ORG:ORG'], 'platform': ['WORK_OF_ART:PRODUCT'], 'tributary': ['LOC:LOC'], 'winner': ['EVENT:PERSON'], 'said to be the same as': ['PERSON:PERSON'], 'composer': ['WORK_OF_ART:PERSON'], 'league': ['ORG:ORG'], 'record label': ['WORK_OF_ART:ORG'], 'distributor': ['WORK_OF_ART:ORG'], 'screenwriter': ['WORK_OF_ART:PERSON'], 'sports season of league or competition': ['DATE:ORG'], 'taxon rank': ['ORG:ORG'], 'location': ['EVENT:GPE'], 'field of work': ['PERSON:ORG'], 'language of work or name': ['WORK_OF_ART:NORP'], 'applies to jurisdiction': ['ORG:GPE'], 'notable work': ['PERSON:WORK_OF_ART'], 'located in the administrative territorial entity': ['GPE:GPE'], 'crosses': ['FAC:LOC'], 'original language of film or TV show': ['WORK_OF_ART:LANGUAGE'], 'competition class': ['PERSON:PRODUCT'], 'part of': ['ORG:ORG'], 'sport': ['PERSON:ORG'], 'constellation': ['LOC:LOC'], 'position played on team / speciality': ['PERSON:PERSON'], 'located in or next to body of water': ['LOC:LOC'], 'voice type': ['PERSON:WORK_OF_ART'], 'follows': ['WORK_OF_ART:WORK_OF_ART'], 'spouse': ['PERSON:PERSON'], 'military rank': ['PERSON:ORG'], 'mother': ['PERSON:PERSON'], 'member of': ['PERSON:ORG'], 'child': ['PERSON:PERSON'], 'main subject': ['WORK_OF_ART:EVENT']}
    
    
    conditions=[] # 收集所有带有relation-contraint的type pair;
    for i,j in rules.items():
        conditions.extend(j)

    clf2 = NLIRelationClassifierWithMappingHead(
        labels=labels,
        template_mapping=template_mapping,
        pretrained_model=config['model_path'],
        valid_conditions=rules,
        negative_threshold=0
    )


    i_num_batch = math.ceil(len(sentences)/sents_batch_size)

    


    start_i=0
    
    end_i=i_num_batch
    for i in tqdm(range(start_i,end_i)):

        # print(i)
        sent_output=[]
        sent_output_nr=[]
        sents_i = sentences[i*sents_batch_size:(i+1)*sents_batch_size]
        subjs_i = subjs[i*sents_batch_size:(i+1)*sents_batch_size]
        objs_i = objs[i*sents_batch_size:(i+1)*sents_batch_size]
        label_i=rels[i*sents_batch_size:(i+1)*sents_batch_size]
        stp_i=subjs_type[i*sents_batch_size:(i+1)*sents_batch_size]
        otp_i=objs_type[i*sents_batch_size:(i+1)*sents_batch_size]
        index_i=index[i*sents_batch_size:(i+1)*sents_batch_size]

        features=[]
        for k0,(t,h,j,k,m,n,id) in enumerate(zip(sents_i,subjs_i,objs_i,label_i,stp_i,otp_i,index_i)):

            # ATTENTION
            # 判断 该sample的 type pair 有没有对应 relation_constraint, 如果没有设置为"all" 也就是不限制relation_constraint(validation condition)
            if str(m)+':'+str(n) in conditions:
                features.append(REInputFeatures(
                    index=id,
                    subj=h,
                    obj=j,
                    pair_type=m+':'+n,
                    context=t,
                    label=k,
                ))
            else:
                features.append(REInputFeatures(
                    index=id,
                    subj=h,
                    obj=j,
                    pair_type='all',
                    context=t,
                    label=k,
                ))
        predictions_2=clf2.predict(features, return_confidences=True, topk=3,batch_size=1)
        top1_pre = [item[0][0] for item in predictions_2]
        gt = [item.label for item in features]
        acc = sum([1 if pre==gt else 0 for pre,gt in zip(top1_pre,gt)])/len(gt)
        print("top1 acc = ",acc)
        for out,feature in zip(predictions_2,features):
            sent_output.append([feature.index, out, feature.label, feature.label])

        
        print("len:     {}".format(sent_output.__len__()))
    # save(sent_output,config['save_dir']+'{}.pkl'.format(i))
   

if __name__ == "__main__":

    TE()
