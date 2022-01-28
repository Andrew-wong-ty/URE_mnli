import time
import os

task_name = "retac_train"

a2t_path = "/home/tywang/myURE/Ask2Transformers"
tac_data_path = "/home/tywang/myURE/URE/TACRED/tacred_norel"  # 名下有 train/dev/test.json
model_path = "/data/transformers/microsoft_deberta-v2-xlarge-mnli"
out_save_path = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out"
cuda_index = 3
current_time=time.strftime("%m-%d-%H%M%S", time.localtime())# 记录被初始化的时间
seed = 16
config_path = "/home/tywang/myURE/URE_mnli/relation_classification/configs/config_tac_partial_constrain.json"  # run evaluation 的config path




# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/tac_xlarge_dev_22631.pkl"
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num226_01-13-222512_.pkl"
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/tac_xlarge_test_15509.pkl"
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num13418_01-14-033752_retac_test.pkl"
run_evaluation_path = os.path.join(tac_data_path,"dev.json")  # tac test
# run_evaluation_path = "/home/tywang/myURE/URE/Re-TACRED/reTAC-output/test.json"  # retac test
outputs = None # 在跑dev set的时候要把output设置为None


split_path = "/home/tywang/myURE/Ask2Transformers/resources/tacred_splits/dev/dev.0.01.split.txt"  # oscar作者本人选出的0.01的split
split = True  # 在数据中(eg dev.json) 随机选择一定比例的数据来跑

################### bool  ##################
get_optimal_threshold = True # 是否计算该数据集对应的optimal threshold
save_dataset_name = "dev_tac_0.01_top123.pkl" # "test.pkl"
 
selected_ratio = 0.01  #0.01
