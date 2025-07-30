import pandas as pd

from hierarchical_content_taxonomy.taxonomy_creation.wordpress_scraping import WordPressScraper
from hierarchical_content_taxonomy.taxonomy_creation.tag_naming.simple import SimpleTagNamer
from hierarchical_content_taxonomy.taxonomy_creation.cluster_creation import ClusterCreator
from hierarchical_content_taxonomy.taxonomy_classification.hierarchical_classifier import HierarchicalClassifier

class HierarchicalTaxonomy:
    def __init__(self, urls: list[str]):
        self.data = None
        self.max_level = None
        self.tag_columns = None
        self.urls = urls
        self.tag_namer = SimpleTagNamer
        self.cluster_creator = ClusterCreator
        self.data_puller = WordPressScraper(urls)

    def get_data(self):
        if self.data is None:
            raise ValueError("Data not set. Please run create_taxonomy() first.")
        return self.data

    def save_data(self, filename):
        if self.data is None:
            raise ValueError("Data not set. Please run create_taxonomy() first.")
        self.data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")

    def pull_data(self):
        scraper = WordPressScraper(self.urls)
        data = scraper.get_wordpress_data('cleaned_wordpress_data.csv')
        self.data = data
        return data

    def generate_cluster_plots(self):
        self.cluster_creator = ClusterCreator(self.data)
        self.cluster_creator.get_cluster_info()
        ## take in user input for cluster distances eventually

    def create_clusters(self, cluster_distances):
        self.cluster_creator.create_cluster_assignments(cluster_distances)
        self.num_levels = self.cluster_creator.n_clusters

    # Not complete yet
    def name_clusters(self):
        tag_namer = self.tag_namer(self.data, self.num_levels)
        self.data = tag_namer.generate_tag_names()
        return self.data

    def classify_taxonomy(self):
        classifier = HierarchicalClassifier(self.data, num_levels=self.num_levels)
        self.classifier = classifier.fit()
        scores = classifier.score()
        return scores
    
    def review_taxonomy(self, data=None) -> pd.DataFrame:
        self.set_tag_level_columns()
        if data is None:
            data = self.data.copy()
        cluster_pivot = pd.pivot_table(data[self.cluster_columns + self.tag_name_columns + ['url']], index=self.cluster_columns + self.tag_name_columns, aggfunc=pd.Series.nunique)
        cluster_pivot = cluster_pivot.sort_values(self.cluster_columns, ascending=True)
        cluster_pivot = cluster_pivot.rename(columns={'url': 'url_count'})
        self.cluster_pivot = cluster_pivot
        return cluster_pivot

    def review_taxonomy_subset(self, level=1, cluster_ids=[1]) -> pd.DataFrame:
        data = self.data.copy()
        subset = data[data[f'topic_level_{level}_cluster_id'].isin(cluster_ids)]
        if subset.empty:
            print(f"No data found for level {level} with cluster IDs {cluster_ids}.")
            return pd.DataFrame()
        return self.review_taxonomy(subset)
    
    def view_titles_in_cluster(self, level=1, cluster_id=1) -> pd.DataFrame:
        data = self.data.copy()
        titles = data[data[f'topic_level_{level}_cluster_id'] == cluster_id]['title']
        cluster_name = data[data[f'topic_level_{level}_cluster_id'] == cluster_id][f'topic_level_{level}'].iloc[0]
        if titles.empty:
            print(f"No titles found for level {level} with cluster ID {cluster_id}.")
            return pd.DataFrame()
        print(f"Titles for level {level}, cluster ID {cluster_id} ({cluster_name}):")
        return titles.reset_index(drop=True)

    ## transform to a general function that takes in a dataframe and returns a pivot table
    def set_tag_level_columns(self) -> None:
        cluster_columns = [f'topic_level_{i+1}_cluster_id' for i in range(self.num_levels)]
        tag_name_columns = [f'topic_level_{i+1}' for i in range(self.num_levels)]
        self.cluster_columns = cluster_columns
        self.tag_name_columns = tag_name_columns
        return None