# 第1章1.4小节代码
#导入包
from collections import defaultdict
from surprise import KNNBaseline
from surprise import Dataset
from surprise import accuracy

def get_top_n(predictions, n):
    return predictions[n]

#1. 数据准备
#使用Movielens 100k数据集中的评分表
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()
#2. 设计推荐方法
#采用协同过滤方法KNNBaseline并设置参数
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
#3. 训练模型
algo.fit(trainset)
#4. 模型预测结果的评估
predictions = algo.test(testset)
accuracy.rmse(predictions)

top_n = get_top_n(predictions, n=10)
#输出推荐结果
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

