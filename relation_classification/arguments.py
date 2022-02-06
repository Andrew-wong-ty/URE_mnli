import time
import os



a2t_path = "/home/tywang/myURE/Ask2Transformers"
tac_data_path = "/home/tywang/myURE/URE/TACRED/tacred_norel"  # 名下有 train/dev/test.json
model_path =  "/data/transformers/microsoft_deberta-v2-xlarge-mnli"
dict_path = "/data/tywang/O2U_model/fine_mnli_only_neg_Label_acc_v1-0.9674.pt" # /data/tywang/O2U_model/fine_mnli_only_pos_acc_v1-0.9926.pt 
load_dict = True                                                         # /data/tywang/O2U_model/fine_mnli_only_neg_acc_v1-0.9601.pt
                                                                         # /data/tywang/O2U_model/fine_mnli_acc_v1-0.9767.pt
out_save_path = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out"
cuda_index = 3
current_time=time.strftime("%m-%d-%H%M%S", time.localtime())# 记录被初始化的时间
seed = 13
config_path = "/home/tywang/myURE/URE_mnli/relation_classification/configs/config_tac_partial_constrain.json"  # run evaluation 的config path



outputs = None # 需要计算mnli的时候, 在跑dev set的时候要把output设置为None

# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/tac_xlarge_dev_22631.pkl"
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num226_01-13-222512_.pkl"
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/tac_xlarge_test_15509.pkl"
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num13418_01-14-033752_retac_test.pkl"
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-06-003328_tac_test_ok_v1_.pkl"  # 一半pos一半neg finetune
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-06-111951_tac_test_ok_v1_.pkl"  # 全pos
# outputs = "/home/tywang/myURE/URE_mnli/relation_classification/mnli_out/num15509_02-06-133146_tac_test_ok_v1_.pkl"  # 全neg

run_evaluation_path = os.path.join(tac_data_path,"test.json")  # tac test
# run_evaluation_path = "/home/tywang/myURE/URE/Re-TACRED/reTAC-output/test.json"  # retac test

task_name = "tac_test_ok_v1_"
split_path = "/home/tywang/myURE/Ask2Transformers/resources/tacred_splits/dev/dev.0.01.split.txt"  # oscar作者本人选出的0.01的split
split = False  # 在数据中(eg dev.json) 随机选择一定比例的数据来跑 加载作者的数据

################### bool  ##################
get_optimal_threshold = True # 是否计算该数据集对应的optimal threshold
save_dataset_name = None # "test.pkl"  如果不save就写None
 
selected_ratio =None  #0.01 # 不select就写None  随机自选
