import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

class BiasSvd(object):
    def __init__(self, alpha, reg_p, reg_q, reg_cu, reg_ci, number_LatentFactors=10, number_epochs=10,
                 columns=["userId", "movieId", "rating"]):
        self.alpha = alpha    # 学习率
        self.reg_p = reg_p    # 用户向量
        self.reg_q = reg_q    # 项目向量
        self.reg_cu = reg_cu  # 用户偏置系数
        self.reg_ci = reg_ci   # 项目偏置系数
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs # 运行轮次
        self.columns = columns

    def fit(self, dataset, valset):
        '''
        fit dataset
        :param dataset: uid, iid, rating
        :return:
        '''

        self.dataset = pd.DataFrame(dataset)
        self.valset = valset

        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        self.globalMean = self.dataset[self.columns[2]].mean()

        self.P, self.Q, self.cu, self.ci, self.Y = self.sgd()

    def _init_matrix(self):
        '''
        初始化P和Q矩阵，同时为设置0，1之间的随机值作为初始值
        :return:
        '''
        # User-LF
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    def predict(self, uid, iid):

        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = self.P[uid]
        q_i = self.Q[iid]
        Y = self.Y

        _sum_yj = np.zeros([1, self.number_LatentFactors])
        jids = self.users_ratings.loc[uid]['movieId'][0]
        Nu = len(jids)
        for jid in jids:
            _sum_yj += Y[jid]

        return self.globalMean + self.cu[uid] + self.ci[iid] + np.dot(p_u + np.sqrt(1 / Nu) * _sum_yj, q_i)

    def test(self, testset):
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating

    def accuracy(self, predict_results):
        def rmse_mae(predict_results):
            '''
            rmse和mae评估指标
            :param predict_results:
            :return: rmse, mae
            '''
            length = 1
            _rmse_sum = 0
            _mae_sum = 0
            for uid, iid, real_rating, pred_rating in predict_results:
                length += 1
                _rmse_sum += (pred_rating - real_rating) ** 2
                _mae_sum += abs(pred_rating - real_rating)
            return np.sqrt(_rmse_sum / length), _mae_sum / length

        return rmse_mae(predict_results)

    def sgd(self):
        '''
        使用随机梯度下降，优化结果
        :return:
        '''
        P, Q = self._init_matrix()

        # 初始化cu、ci的值，全部设为0
        cu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        ci = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))
        Y = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))

        rmse_list = []
        mae_list = []

        for i in range(self.number_epochs):
            print("iter%d" % i)
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):

                jids = self.users_ratings.loc[uid]['movieId'][0]
                Nu = len(jids)
                _sum_yj = np.zeros([self.number_LatentFactors])

                for jid in jids:
                    _sum_yj += Y[jid]

                # sum_v_yj =

                v_pu = P[uid]
                v_qi = Q[iid]
                err = np.float32(
                    r_ui - self.globalMean - cu[uid] - ci[iid] - np.dot(v_pu + np.sqrt(1 / Nu) * _sum_yj, v_qi))
                for jid in jids:
                    Y[jid] += self.alpha * (err * np.sqrt(1 / Nu) * v_qi - 0.01 * Y[jid])

                P[uid] += self.alpha * (err * v_qi - self.reg_p * v_pu)
                Q[iid] += self.alpha * (err * (v_pu + np.sqrt(1 / Nu) * _sum_yj) - self.reg_q * v_qi)

                cu[uid] += self.alpha * (err - self.reg_cu * cu[uid])
                ci[iid] += self.alpha * (err - self.reg_ci * ci[iid])

                error_list.append(err ** 2)
            print(np.sqrt(np.mean(error_list)))
            self.P = P
            self.Q = Q
            self.cu = cu
            self.ci = ci
            self.Y = Y

            pred_results = self.test(self.valset)
            rmse, mae = self.accuracy(pred_results)
            rmse_list.append(rmse)
            mae_list.append(mae)
            print("rmse: ", rmse, "mae: ", mae)

        x = range(1, self.number_epochs + 1)
        plt.plot(x, rmse_list)
        plt.title('SVDpp_SGD')
        plt.xlabel('epoch')
        plt.ylabel('RMSE')
        plt.show()
        return P, Q, cu, ci, Y

def predict_ratings(user_id, trainset_user_item_matrix, user_similarity_df, valset_user_item_matrix):
    """
    预测指定用户对所有电影的评分。
    :param user_id: 用户ID
    :param trainset_user_item_matrix: 用户-物品评分矩阵
    :param user_similarity_df: 用户相似度矩阵
    :return: 预测评分
    """
    # 获取目标用户的评分
    trainset_user_ratings = trainset_user_item_matrix.loc[user_id]
    valset_user_ratings = trainset_user_item_matrix.loc[user_id]

    # 找到与目标用户相似的用户及其相似度
    similar_users = user_similarity_df[user_id].drop(user_id)
    similar_users = similar_users[similar_users > 0]  # 筛选出相似度大于0的用户

    # 计算预测评分
    pred_ratings = trainset_user_ratings.copy()
    length = 1
    _rmse_sum = 0
    _mae_sum = 0
    for movie_id in trainset_user_item_matrix.columns:
        if trainset_user_ratings[movie_id] == 0:  # 只预测未评分的电影
            numerator = 0
            denominator = 0
            for other_user_id, similarity in similar_users.items():
                other_user_rating = trainset_user_item_matrix.at[other_user_id, movie_id]
                if other_user_rating > 0:  # 其他用户对该电影的评分
                    numerator += similarity * other_user_rating
                    denominator += abs(similarity)
            pred_ratings[movie_id] = numerator / denominator if denominator != 0 else 0
            length += 1
            _rmse_sum += (pred_ratings[movie_id] - valset_user_ratings[movie_id]) ** 2
            _mae_sum += abs(pred_ratings[movie_id] - valset_user_item_matrix.at[user_id, movie_id])
    return np.sqrt(_rmse_sum / length), _mae_sum / length


if __name__ == '__main__':
    # ==================== 方法选择配置 ====================
    # 可选值: 'svdpp' 或 'user_cf'
    # 'svdpp': 使用SVD++算法
    # 'user_cf': 使用基于用户的协同过滤方法
    METHOD = 'svdpp'
    # =====================================================

    trainset = pd.read_csv('ml-100k\\u1.base', sep='\t', names=["userId", "movieId", "rating"], usecols=range(3))
    valset = pd.read_csv('ml-100k\\u1.test', sep='\t', names=["userId", "movieId", "rating"], usecols=range(3))

    if METHOD == 'svdpp':
        # ============== SVD++ 方法 ==============
        algo = BiasSvd(0.01, 0.01, 0.01, 0.01, 0.01, 1, 40)
        algo.fit(trainset, valset)

        # print(trainset[trainset["userId"] == 1])
        # print( len(bsvd.users_ratings.loc[1]['movieId'][0]) )
        pred_results = algo.test(valset)

        rmse, mae = algo.accuracy(pred_results)

        print("rmse: ", rmse, "mae: ", mae)

    elif METHOD == 'user_cf':
        # ============== 基于用户的协同过滤方法 ==============
        # 构建用户-物品矩阵
        trainset_user_item_matrix = trainset.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
        valset_user_item_matrix = valset.pivot_table(index='userId', columns='movieId', values='rating', fill_value=0)
        
        # 统一训练集和验证集的行(用户ID)和列(电影ID),确保两个矩阵包含相同的用户和电影集合
        all_user_ids = trainset_user_item_matrix.index.union(valset_user_item_matrix.index)
        all_movie_ids = trainset_user_item_matrix.columns.union(valset_user_item_matrix.columns)
        trainset_user_item_matrix = trainset_user_item_matrix.reindex(index=all_user_ids, columns=all_movie_ids, fill_value=0)
        valset_user_item_matrix = valset_user_item_matrix.reindex(index=all_user_ids, columns=all_movie_ids, fill_value=0)

        # 计算用户之间的余弦相似度
        user_similarity = cosine_similarity(trainset_user_item_matrix)
        user_similarity_df = pd.DataFrame(user_similarity, index=trainset_user_item_matrix.index, columns=trainset_user_item_matrix.index)

        from joblib import Parallel, delayed
        from tqdm import tqdm

        # 定义一个计算单个用户的函数
        def compute_user_metrics(user_id):
            return predict_ratings(user_id, trainset_user_item_matrix, user_similarity_df, valset_user_item_matrix)


        # 并行计算
        results = Parallel(n_jobs=-1)(
            delayed(compute_user_metrics)(user_id) for user_id in tqdm(range(1, 944), desc='user'))

        # 累加结果
        rmse = sum(r[0] for r in results)
        mae = sum(r[1] for r in results)

        print(f"Total RMSE: {rmse/943}, Total MAE: {mae/943}")

    else:
        raise ValueError(f"不支持的方法: {METHOD}，请选择 'svdpp' 或 'user_cf'")