from random import random
import time
import os
import json
import random


# 公用
#  "/data/transformers/microsoft_deberta-v2-xlarge-mnli"
#  "/data/transformers/microsoft_deberta-v2-xxlarge-mnli"  
model_path = "/data/transformers/microsoft_deberta-v2-xlarge-mnli"
current_time=time.strftime("%m-%d-%H%M%S", time.localtime())# 记录被初始化的时间
cuda_index = 0
seed = 16
a2t_path = "/home/tywang/myURE/Ask2Transformers"
dict_path = "/data/tywang/O2U_model/fine_mnli_wiki_random_xlarge_epo6-0.9827.pt" # "/data/tywang/O2U_model/fine_mnli_only_neg_Label_NEW_NEG_TEMPLATE_acc_v1-0.9779.pt" # /data/tywang/O2U_model/fine_mnli_only_pos_acc_v1-0.9926.pt 
load_dict = True                                                         # /data/tywang/O2U_model/fine_mnli_only_neg_acc_v1-0.9601.pt
                                                                         # /data/tywang/O2U_model/fine_mnli_acc_v1-0.9767.pt
################### bool  ##################
get_optimal_threshold = False # 是否计算该数据集对应的optimal threshold
default_optimal_threshold = 0.993993993993994
save_dataset_name = None # "test.pkl"  如果不save就写None

selected_ratio =None #0.01 # 不select就写None  随机自选

task_name = "wiki_test_finetune"





"""
tac 专用
"""

# dataset = "tac"
# out_save_path = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out"

# config_path = "/home/tywang/myURE/URE_mnli/relation_classification/configs/config_tac_partial_constrain.json"  # run evaluation 的config path
# label2id_path = "/home/tywang/myURE/URE/O2U_bert/tac_data/whole/rel2id.pkl"


# outputs = None # 需要计算mnli的时候, 在跑dev set的时候要把output设置为None
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-12-184928_WholeProc0.05_NEUTRAL.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-10-221227_WholeProc0.1.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-10-160724_tac_test_seed13_no_finetune.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-09-142635_0.01train.pkl"
# # outputs ="/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/time_12-10-012219_model_-v2-xlarge-mnli_nFea_68124_output.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/tac_xlarge_dev_22631.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/tac_xlarge_train_68124.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-10-151430_tac_test_seed13_no_finetune.pkl"  # tac test ori
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num13418_01-14-033752_retac_test.pkl"
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-06-003328_tac_test_ok_v1_.pkl"  # 一半pos一半neg finetune
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-06-111951_tac_test_ok_v1_.pkl"  # 全pos
# # outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-06-133146_tac_test_ok_v1_.pkl"  # 全neg

# mode = "test"
# tac_data_path = "/home/tywang/myURE/URE/TACRED/tacred_norel"  # 名下有 train/dev/test.json
# run_evaluation_path = os.path.join(tac_data_path,"{}.json".format(mode))  # tac test
# # run_evaluation_path = "/home/tywang/myURE/URE/Re-TACRED/reTAC-output/test.json"  # retac test


# split_path = "/home/tywang/myURE/Ask2Transformers_old_version/resources/tacred_splits/train/0.01.split.txt"   # "/home/tywang/myURE/Ask2Transformers/resources/tacred_splits/dev/dev.0.01.split.txt"  # oscar作者本人选出的0.01的split
# split = False  # 在数据中(eg dev.json) 随机选择一定比例的数据来跑 加载作者的数据



"""
wiki 专用
"""
dataset = "wiki"
out_save_path = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_wiki_out"
config_path = "/home/tywang/myURE/URE_mnli/relation_classification/configs/config_wiki_partial_constraint.json"
label2id_path = "/home/tywang/myURE/URE/WIKI/typed/label2id.pkl"
outputs = None
outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_wiki_out/num5600_02-16-203550_wiki_test_rerun.pkl"  # 需要自己跑
generate_data = False
mode = "test"
wiki_data_path = "/home/tywang/myURE/URE/WIKI/typed"  # 名下有 train/dev/test 的数据
run_evaluation_path = os.path.join(wiki_data_path,"wiki_{}withtype_premnil.pkl".format(mode))  # tac test
get_optimal_threshold = False
default_optimal_threshold = 0
