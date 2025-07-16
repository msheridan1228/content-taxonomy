import pandas as pd

from hierarchical_content_taxonomy.taxonomy_creation.wordpress_scraping import WordPressScraper
from hierarchical_content_taxonomy.taxonomy_creation.tag_naming.namer import TagNamer
from hierarchical_content_taxonomy.taxonomy_creation.cluster_creation import ClusterCreator
from hierarchical_content_taxonomy.taxonomy_classification.hierarchical_classifier import HierarchicalClassifier

class HierarchicalTaxonomy:
    def __init__(self, urls: list[str]):
        self.data = None
        self.max_level = None
        self.tag_columns = None
        self.urls = urls
        self.tag_namer = TagNamer
        self.cluster_creator = ClusterCreator
        self.data_puller = WordPressScraper(urls)

    def create_taxonomy(self):
        self.data = self.pull_data()
        if self.data.empty:
            print("No data to create taxonomy.")
            return None

        self.cluster_creator = ClusterCreator(self.data)
        self.cluster_creator.get_cluster_info()
        self.data = self.cluster_creator.create_cluster_assignments(cluster_distances=[0.5, 1.0, 1.5])
        
        self.cluster_namer = TagNamer(self.data)
        self.data = self.cluster_namer.name_clusters(self.data)

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

    ## Not complete yet
    # def name_clusters(self):
    #     tag_namer = TagNamer(self.data)
    #     tag_namer.fit_predict()
    #     self.tag_columns = tag_namer.tag_columns
    #     self.max_level = tag_namer.get_max_level()
    #     self.data = tag_namer.data
    #     return self.data
    
    def classify_taxonomy(self, model_path):
        classifier = HierarchicalClassifier(model_path, self.data, self.tag_columns)
        classifier.fit()
        predictions = classifier.predict()
        self.data = predictions
        return self.data

    def set_tag_columns(self, tag_columns):
        if isinstance(tag_columns, list):
            self.tag_columns = tag_columns
        else:
            raise ValueError("Tag columns must be a list of strings representing the column names.")

    def set_levels(self, levels):
        if isinstance(levels, list):
            self.levels = levels
        else:
            raise ValueError("Levels must be a list of strings representing the hierarchy levels.")

    def get_max_level(self):
        return self.max_level

    def get_top_level_categories(self):
        if self.data is None:
            raise ValueError("Data not set. Please set the data before getting top-level categories.")
        return self.data['top_level_category'].unique().tolist()
    
    def get_subcategories(self, top_level_category):
        if self.data is None:
            raise ValueError("Data not set. Please set the data before getting subcategories.")
        return self.data[self.data['top_level_category'] == top_level_category]['subcategory'].unique().tolist()
    
    def get_parent_category(self, subcategory):
        if self.data is None:
            raise ValueError("Data not set. Please set the data before getting parent category.")
        parent = self.data[self.data['subcategory'] == subcategory]['top_level_category']
        if not parent.empty:
            return parent.iloc[0]
        else:
            raise ValueError(f"No parent category found for subcategory: {subcategory}")   
