# 【例3-1】
#导入CountVectorizer包
from sklearn.feature_extraction.text import CountVectorizer
#声明字符串类型的数组
test_text=['Most people watch TV to get news','Most people get news by watching TV']
#实例化CountVectorizer类
vectorizer = CountVectorizer()
#统计词频并生成向量（矩阵）
result=vectorizer.fit_transform(test_text).toarray()
#输出转换结果
print(result)
# 【例3-2】
from sklearn.feature_extraction.text import TfidfVectorizer
test_text=['Most people watch TV to get news','Most people get news by watching TV']
#创建TfidfVectorizer的实例
vectorizer = TfidfVectorizer()
#将给定的文本分别计算词汇的tfidf值，并将结果转换为数组(矩阵形式)
matrix = vectorizer.fit_transform(test_text).toarray()
#打印文本中的特征词
print("feature_names:",vectorizer.get_feature_names())
#打印文本的向量表示结果（矩阵形式）
print("matrix:\n",matrix)

# 【例3-3】
# 导入gensim库
from gensim.models import word2vec
sentences = ["Most people watch TV to get news","Most people get news by watching TV"]
sentences= [s.split() for s in sentences]
print(sentences)
model = word2vec.Word2Vec(sentences, min_count=1)
model.save('model')
#加载模型
model = word2vec.Word2Vec.load('model')
#获取“watch”的词向量
word_vector=model.wv['watch']
print(word_vector)

# 【例3-4】
import jieba
text = "今天是星期一,星期一是今天,星期一是一个新的开始"
#精确模式
print("/".join(jieba.cut(text,cut_all=False)))
#全模式
print("/".join(jieba.cut(text, cut_all=True)))
#搜索引擎模式
print("/".join(jieba.cut_for_search(text)))
# 【例3-5】
import jieba
from sklearn.feature_extraction.text import CountVectorizer
text = ["今天是星期一","星期一是今天","星期一是一个新的开始"]
#使用精确模式分词
#使用data保存分词结果
data=list()
for t in text:
    data.append(' '.join(list(jieba.cut(t))))
#打印分词后的结果
print("data",data)
# 使用CountVectorizer进行词频向量表示
count_vectorizer = CountVectorizer()
count_vectors = count_vectorizer.fit_transform(data)
# 输出词频向量的特征名和向量表示
print("CountVectorizer特征名:", count_vectorizer.get_feature_names_out())
print("CountVectorizer向量表示:\n", count_vectors.toarray())
# 使用TfidfVectorizer进行TF-IDF向量表示
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectors = tfidf_vectorizer.fit_transform(data)
# 输出TF-IDF向量的特征名和向量表示
print("TfidfVectorizer特征名:", tfidf_vectorizer.get_feature_names_out())
print("TfidfVectorizer向量表示:\n", tfidf_vectors.toarray())
# 【例3-6】
#导入包
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
#声明字符串类型的数组
test_text=['Monday is Monday','it is Monday today','Monday is a new beginning']
#实例化CountVectorizer()类
vectorizer = CountVectorizer()
#test_text中词频统计结果
result=vectorizer.fit_transform(test_text)
#将文本转换为矩阵
matrix = vectorizer.fit_transform(test_text).toarray()
#输出矩阵
print("matrix:\n",matrix)
#矩阵中的每一行即代表句子的向量,计算每个向量间的余弦相似度
print("cosine_simliarity:\n",cosine_similarity(matrix))
# 【例3-7】
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# 读取电影数据,忽略编码错误的数据
movies = pd.read_csv('movies.dat', sep='::',header=None, names=['movie_id', 'title', 'genres'],encoding_errors="ignore")
# 提取电影名称特征
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['title'])
# 计算电影之间的余弦相似度
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# 创建电影名称到索引的映射
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
# 定义电影推荐函数
def get_recommendations(title, cosine_sim=cosine_sim):
    # 获取电影的索引
    idx = indices[title]
    # 获取该电影与其他所有电影的相似度得分
    sim_scores = list(enumerate(cosine_sim[idx]))
    # 按相似度得分排序
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # 获取前 5个相似的电影（不包括自身）
    sim_scores = sim_scores[1:6]
    # 获取相似电影的索引
    movie_indices = [i[0] for i in sim_scores]

#根据电影名称进行推荐
movie_title = input('请输入电影名称：')
recommended_movies = get_recommendations(movie_title)
print(f"基于电影《{movie_title}》的推荐电影：")
print(recommended_movies)
# 【例3-8】
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from collections import defaultdict
# 加载数据
movies = pd.read_csv('movies.dat', sep='::',  header=None,
                     names=['movie_id', 'title', 'genres'], encoding='ISO-8859-1')
ratings = pd.read_csv('ratings.dat', sep='::',  header=None,
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

#将电影名称转换为 TF-IDF 向量
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['title'])
#计算电影之间的余弦相似度
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
# 创建电影 ID 到索引的映射
indices = pd.Series(movies.index, index=movies['movie_id'])
# 划分训练集和测试集
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
# 为每个用户生成推荐列表
def get_user_recommendations(user_id, train_data, cosine_sim, top_n=5):
    # 获取用户在训练集中观看过的电影
    user_movies = train_data[train_data['user_id'] == user_id]['movie_id'].tolist()
    recommendation_scores = defaultdict(float)
    # 遍历用户观看过的每一部电影
    for movie_id in user_movies:
        idx = indices[movie_id]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [score for score in sim_scores if movies.iloc[score[0]]['movie_id'] not in user_movies]
        for score in sim_scores:
            recommendation_scores[movies.iloc[score[0]]['movie_id']] += score[1]
    # 按推荐得分排序
    sorted_scores = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)
    # 获取前 top_n 个推荐电影的 ID
    top_movie_ids = [i[0] for i in sorted_scores[:top_n]]
    return top_movie_ids

# 计算准确率和召回率
def calculate_metrics(train_data, test_data, cosine_sim, top_n=5):
    all_users = test_data['user_id'].unique()[:10]
    total_relevant = 0
    total_recommended = 0
    total_hit = 0
    for user_id in all_users:
        # 生成推荐列表
        recommended_movies = get_user_recommendations(user_id, train_data, cosine_sim, top_n)
        # 获取用户在测试集中实际观看过的电影
        relevant_movies = test_data[test_data['user_id'] == user_id]['movie_id'].tolist()
        total_relevant += len(relevant_movies)
        total_recommended += len(recommended_movies)
        # 计算命中的电影数量
        hit_count = len(set(recommended_movies).intersection(set(relevant_movies)))
        total_hit += hit_count
    # 计算准确率和召回率
    precision = total_hit / total_recommended if total_recommended > 0 else 0
    recall = total_hit / total_relevant if total_relevant > 0 else 0
    return precision, recall
# 计算准确率和召回率
precision, recall = calculate_metrics(train_data, test_data, cosine_sim, top_n=5)
print(f"准确率: {precision}")
print(f"召回率: {recall}")
