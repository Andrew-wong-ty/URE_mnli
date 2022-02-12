import numpy as np
from sklearn.metrics import precision_recall_fscore_support
labels2id = np.load("/home/tywang/myURE/URE_mnli/relation_classification/why/labels2id.npy",allow_pickle=True) # no_relation 是0
# applied_threshold_output 是每一个句子对于每个relation的概率, shape=(15509,42)
applied_threshold_output = np.load("/home/tywang/myURE/URE_mnli/relation_classification/why/applied_threshold_output.npy")

labels = np.load("/home/tywang/myURE/URE_mnli/relation_classification/why/labels.npy")


# 尝试一下两种方法计算prediction
pre_1 = applied_threshold_output.argmax(-1)
pre_2 = np.argsort(applied_threshold_output)[:, ::-1][:, 0]
res = all(pre_1==pre_2)  # 发现不一样?

#以下是分别用pre_1和pre_2算 评价指标
acc1 = sum(labels==pre_1)/len(pre_1)  # 这里的f1就和我给你的一样
precision_1, recall_1, f1_1, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
    labels, pre_1, average="micro", labels=list(range(1, 42)) # 0是no_relation
)

acc2 = sum(labels==pre_2)/len(pre_2)
precision_2, recall_2, f1_2, _ = precision_recall_fscore_support(  # 应该是只算pos的,  因为当预测全为neg_rel的时候, f1 = 0
    labels, pre_2, average="micro", labels=list(range(1, 42)) # 0是no_relation
)




debug_stop = 1