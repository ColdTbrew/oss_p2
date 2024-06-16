import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix

# 데이터 로드 
ratings = pd.read_csv('/mnt/f/oss_p2/ml-1m/ratings.dat', sep='::', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'], engine='python')

ratings.drop(['Timestamp'], axis=1, inplace=True)

rating_pivot = ratings.pivot(index='UserID', columns='MovieID', values='Rating').fillna(0)

## 1st KMeans clustering

# NumPy 배열로 변환
user_item_matrix_np = rating_pivot.to_numpy()

# K-Means 클러스터링
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(user_item_matrix_np)

# 클러스터 레이블 출력
labels = kmeans.labels_

# 각 사용자별 클러스터 레이블을 데이터프레임으로 저장
user_clusters = pd.DataFrame({'UserID': rating_pivot.index, 'Cluster': labels})

print(user_clusters)

# 클러스터 레이블을 원본 데이터프레임에 추가
ratings = ratings.merge(user_clusters, on='UserID')

# 각 클러스터별 통계 출력
print(ratings.groupby('Cluster').mean())




## 2nd Group Recommender Algorithms

# 클러스터가 1인것 만 선택
cluster1 = ratings[ratings['Cluster'] == 0]
cluster1_pivot = cluster1.pivot(index='UserID', columns='MovieID', values='Rating')
print(cluster1_pivot.shape)
# 클러스터가 2인것 만 선택
cluster2 = ratings[ratings['Cluster'] == 1]
cluster2_pivot = cluster2.pivot(index='UserID', columns='MovieID', values='Rating')
print(cluster2_pivot.shape)
# 클러스터가 3인것 만 선택
cluster3 = ratings[ratings['Cluster'] == 2]
cluster3_pivot = cluster3.pivot(index='UserID', columns='MovieID', values='Rating')
print(cluster3_pivot.shape)

# 알고리즘별 추천 결과 출력 함수 정의
def print_reco_by_algo(cluster):
    cluster_pivot_AU = cluster.sum(axis=0).sort_values(ascending=False).head(10)
    print("**Additive Utilitarian : \n", cluster_pivot_AU)
    cluster_pivot_AVG = cluster.mean(axis=0).sort_values(ascending=False).head(10)
    print("**Average : \n", cluster_pivot_AVG)
    cluster_pivot_SC = cluster[cluster > 0].count(axis=0).sort_values(ascending=False).head(10)
    print("**Simple Count : \n", cluster_pivot_SC)
    cluster_pivot_AV = cluster[cluster > 4].count(axis=0).sort_values(ascending=False).head(10)
    print("**Approval Voting : \n", cluster_pivot_AV)

    ranks = cluster.rank(axis=1, method='average' )-1
    borda_aggregated = ranks.sum(axis=0)
    borda_aggregated_series = pd.Series(borda_aggregated, index=cluster.columns)
    print("**Borda Voting : \n", borda_aggregated_series.sort_values(ascending=False).head(10))
    
    # Copeland Rule 구현
    num_items = cluster.shape[1]

    user_item_matrix_np = cluster.values
    items = cluster.columns
    comparison_matrix = np.zeros((num_items, num_items), dtype=int)

    # 넘파이 벡터연산으로 빠르게 연산 수행, 기존 방법은 매우 느려서 사용 불가
    for i in range(num_items):
        for j in range(i + 1, num_items):
            item1_scores = user_item_matrix_np[:, i]
            item2_scores = user_item_matrix_np[:, j]
            
            item1_wins = np.sum(item1_scores > item2_scores)
            item2_wins = np.sum(item1_scores < item2_scores)
            
            if item1_wins > item2_wins:
                comparison_matrix[i, j] = -1
                comparison_matrix[j, i] = 1
            elif item2_wins > item1_wins:
                comparison_matrix[i, j] = 1
                comparison_matrix[j, i] = -1

    # 결과를 DataFrame으로 변환
    comparison_df = pd.DataFrame(comparison_matrix, index=items, columns=items)
    copeland_rule = comparison_df.sum(axis=0)

    print("**Copeland Rule: \n", copeland_rule.sort_values(ascending=False).head(10))

print("------------Cluster 1------------")
print_reco_by_algo(cluster1_pivot)

print("------------Cluster 2------------")
print_reco_by_algo(cluster2_pivot)

print("------------Cluster 3------------")
print_reco_by_algo(cluster3_pivot)
