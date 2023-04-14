import pandas as pd
import numpy as np

# 加载数据
ratings_data = pd.read_csv('ratings.csv')
movie_names = pd.read_csv('movies.csv')

# 合并数据
movie_data = pd.merge(ratings_data, movie_names, on='movieId')

# 计算每个电影的平均评分
movie_data.groupby('title')['rating'].mean().sort_values(ascending=False).head()

# 创建电影评分矩阵
user_ratings = movie_data.pivot_table(index=['userId'], columns=['title'], values='rating')

# 计算每个电影的评分数量
ratings_count = pd.DataFrame(movie_data.groupby('title')['rating'].count())

# 选择至少有 100 个评分的电影进行推荐
popular_movies = list(ratings_count[ratings_count['rating'] >= 100].index)

# 创建电影相似度矩阵
movie_similarity = user_ratings.corr(method='pearson', min_periods=100)

# 获取用户的观看记录
user_id = 100
user_ratings = user_ratings.loc[user_id].dropna()

# 计算电影推荐得分
recommendation_scores = pd.Series()
for movie in popular_movies:
    if movie in user_ratings:
        continue
    similarity_scores = movie_similarity[movie].dropna()
    similarity_scores = similarity_scores.map(lambda x: x * user_ratings.loc[user_id][similarity_scores.index].mean())
    recommendation_scores = recommendation_scores.append(similarity_scores)

# 推荐前 10 部电影
recommendation_scores = recommendation_scores.groupby(recommendation_scores.index).sum()
recommendation_scores.sort_values(ascending=False, inplace=True)
recommendation_scores.head(10)