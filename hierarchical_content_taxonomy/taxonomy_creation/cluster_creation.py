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
        # save embeddings
        self.docs_df.to_csv('embeddings.csv', index=False)
        print('creating linkage vectors')
        self.create_linkage_matrix()
        print('creating dendrogram')
        self.create_dendrogram()
        print('finding optimal clusters')
        self.find_optimal_clusters()

    def create_cluster_assignments(self, cluster_distances: list[float]) -> pd.DataFrame:
        self.set_cluster_distances(cluster_distances)
        self.assign_cluster_labels()
        return self.docs_df

    def set_embedder(self, embedder):
        self.embedder = embedder

    def create_embeddings(self) -> None:
        self.docs_df['title_plus_text'] = self.docs_df['title'] + '. ' + self.docs_df['text']
        self.embeddings = self.embedder(self.docs_df['title_plus_text']).numpy()
        self.docs_df['text_embedding'] = self.embeddings.tolist()

    def create_linkage_matrix(self) -> None:
        if self.embeddings is None:
            raise ValueError("Embeddings must be created before generating the linkage matrix.")

        self.linkage_matrix = fc.linkage_vector(self.embeddings, method=self.method, metric=self.metric)

    def create_dendrogram(self) -> None:
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix must be created before generating the dendrogram.")

        plt.figure(figsize=(10, 7))
        plt.title("Dendrograms")
        dend = shc.dendrogram(self.linkage_matrix)
        plt.show()

    def find_optimal_clusters(self) -> None:
        if self.linkage_matrix is None:
            raise ValueError("Linkage matrix must be created before finding optimal clusters.")
        
        y_axis = self.linkage_matrix[:, 2]
        distances = np.logspace(np.log10(min(y_axis)), np.log10(max(y_axis)), num=25).tolist()

        distortions = []
        num_clusters = []
        for distance in distances:
            clusters = fcluster(self.linkage_matrix, distance, criterion='distance')
            if len(np.unique(clusters)) < 2:
                print(f"Skipping distance={distance} bc you need at least 2 clusters")
                distances.remove(distance)
                continue
            score = metrics.silhouette_score(self.embeddings, clusters)

            distortions.append(np.round(score, 3))
            num_clusters.append(len(np.unique(clusters)))

        plt.figure(figsize=(12, 8))
        plt.plot(num_clusters, distortions, 'bx-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Avg Silhouette Coefficient')
        plt.title('The Silhouette method showing the optimal number of clusters')
        for x,y,z in zip(num_clusters, distortions, distances):
          label = f"d={np.round(z, 3)},\n n_clust={x}"
          plt.annotate(label, # this is the text
                      (x,y), # this is the point to label
                      textcoords="offset points", # how to position the text
                      xytext=(0,10), # distance from text to points (x,y)
                      ha='center') # horizontal alignment can be left, right or cente
        plt.show()

    def set_cluster_distances(self, cluster_distances: list[float]) -> None:
        if not isinstance(cluster_distances, list):
            raise ValueError("Cluster distances must be a list of floats representing the number of clusters.")
        if not all(isinstance(i, float) for i in cluster_distances):
            raise ValueError("All elements in cluster distances must be floats.")
        if len(cluster_distances) == 0:
            raise ValueError("Cluster distances list cannot be empty.")
        if any(i<=0 or i > 10 for i in cluster_distances):
            raise ValueError("Cluster distances must be between 0 and 10.")
        self.cluster_distances = cluster_distances
        self.n_clusters = len(cluster_distances)
        self.topic_levels = {'topic_level_' + str(i+1): cluster_distances[i] for i in range(self.n_clusters)}

    def get_num_clusters(self) -> int:
        if not hasattr(self, 'n_clusters'):
            raise ValueError("Number of clusters has not been set. Please set cluster distances first.")
        return self.n_clusters

    def assign_cluster_labels(self) -> None:
        if self.embeddings is None:
            raise ValueError("Embeddings must be set before creating taxonomy.")
        if not isinstance(self.n_clusters, int) or self.n_clusters <= 0:
            raise ValueError("Number of clusters must be a positive integer.")
        for key, value in self.topic_levels.items():
            self.docs_df[key + '_cluster_id'] = fcluster(self.linkage_matrix, value, criterion='distance')
  
    def check_required_columns(self, required_columns=['text', 'title']):
        for column in required_columns:
            if column not in self.docs_df.columns:
                raise ValueError(f"Required column '{column}' is missing from the DataFrame.")