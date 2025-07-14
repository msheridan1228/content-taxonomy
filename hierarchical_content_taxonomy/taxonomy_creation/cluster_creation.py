import scipy.cluster.hierarchy as shc
import fastcluster as fc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_hub as hub
from sklearn import metrics
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering 

class ClusterCreator:
    def __init__(self, docs_df: pd.DataFrame, method='ward', metric='euclidean', embed_module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"):
        self.docs_df = docs_df
        self.embedder = hub.load(embed_module_url)
        self.method = method
        self.metric = metric

    def get_cluster_info(self) -> None:
        self.check_required_columns()
        print('creating embeddings')
        self.create_embeddings()
        print('creating linkage vectors')
        self.create_linkage_matrix(self.method, self.metric)
        print('creating dendrogram')
        self.create_dendrogram()
        print('finding optimal clusters')
        self.find_optimal_clusters(self.method, self.metric)

    def create_cluster_assignments(self, cluster_distances: list[int]) -> pd.DataFrame:
        self.set_cluster_distances(cluster_distances)
        self.assign_cluster_labels()
        return self.docs_df

    def set_embedder(self, embedder):
        self.embedder = embedder

    def create_embeddings(self) -> None:
        self.docs_df['title_plus_text'] = self.docs_df['title'] + '. ' + self.docs_df['text']
        self.embeddings = self.embedder(self.docs_df['title_plus_text']).tolist().numpy()
        self.docs_df['text_embedding'] = self.embeddings.tolist()

    def create_linkage_matrix(self) -> None:
        if not self.embeddings:
            raise ValueError("Embeddings must be created before generating the linkage matrix.")

        self.docs_df['linkage_matrix'] = fc.linkage_vector(self.embeddings, method=self.method, metric=self.metric)

    def create_dendrogram(self) -> None:
        if 'linkage_matrix' not in self.docs_df.columns:
            raise ValueError("Linkage matrix must be created before generating the dendrogram.")

        plt.figure(figsize=(10, 7))  
        plt.title("Dendrograms")  
        dend = shc.dendrogram(self.docs_df['linkage_matrix'])
        plt.show()

    def find_optimal_clusters(self) -> None:
        if 'linkage_matrix' not in self.docs_df.columns:
            raise ValueError("Linkage matrix must be created before finding optimal clusters.")
        K = np.unique((np.round(np.logspace(.2, 2.85, num = 25), 0)).astype(int))

        distortions = []
        for k in K:
            clusters = fcluster(self.docs_df['linkage_matrix'], k, criterion='distance')
            distortions.append(metrics.silhouette_score(self.embeddings, clusters))

        plt.figure(figsize=(12, 8))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Avg Silhouette Width')
        plt.xscale('log')
        plt.title('The Silhouette method showing the optimal k')
        for x,y in zip(K,distortions):
          label = "{}".format(x)
          plt.annotate(label, # this is the text
                      (x,y), # this is the point to label
                      textcoords="offset points", # how to position the text
                      xytext=(0,10), # distance from text to points (x,y)
                      ha='center') # horizontal alignment can be left, right or cente
        plt.show()

    def set_cluster_distances(self, cluster_distances: list[int]) -> None:
        if not isinstance(cluster_distances, list):
            raise ValueError("Cluster distances must be a list of integers representing the number of clusters.")
        if not all(isinstance(i, int) for i in cluster_distances):
            raise ValueError("All elements in cluster distances must be integers.")
        if len(cluster_distances) == 0:
            raise ValueError("Cluster distances list cannot be empty.")
        if any(i<=0 or i > 100 for i in cluster_distances):
            raise ValueError("Cluster distances must be between 1 and 100.")
        self.cluster_distances = cluster_distances
        self.n_clusters = len(cluster_distances)
        self.topic_levels = {'topic_level_' + str(i+1): cluster_distances[i] for i in range(self.n_clusters)}

    def assign_cluster_labels(self) -> None:
        if self.docs_df['text_embedding'] is None:
            raise ValueError("Embeddings must be set before creating taxonomy.")
        if not isinstance(self.n_clusters, int) or self.n_clusters <= 0:
            raise ValueError("Number of clusters must be a positive integer.")
        for key, value in self.topic_levels.items():
            self.docs_df[key + '_cluster_id'] = fcluster(self.docs_df['linkage_matrix'], value, criterion='distance')
  
    def check_required_columns(self, required_columns=['text', 'title']):
        for column in required_columns:
            if column not in self.docs_df.columns:
                raise ValueError(f"Required column '{column}' is missing from the DataFrame.")