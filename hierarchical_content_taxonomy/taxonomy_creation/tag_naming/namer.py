from abc import ABC, abstractmethod
import pandas as pd
## Placeholder for now
### Want to instantiate a base class that can be extended for different tag naming methods
# Keep methods consistent using this base class
class TagNamer:
    def __init__(self, data, num_levels):
        self.data = data
        self.num_levels = num_levels
        self.set_tag_level_columns()
        self.required_columns = ['text', 'title', 'url'] + self.cluster_columns
        self.check_required_columns()

    def check_required_columns(self):
        for column in self.required_columns:
            if column not in self.data.columns:
                raise ValueError(f"Required column '{column}' is missing from the DataFrame.")

    @abstractmethod
    def generate_tag_names(self):
        pass
    
    ## transform to a general function that takes in a dataframe and returns a pivot table
    def set_tag_level_columns(self) -> None:
        cluster_columns = [f'topic_level_{i+1}_cluster_id' for i in range(self.num_levels)]
        tag_name_columns = [f'topic_level_{i+1}' for i in range(self.num_levels)]
        self.cluster_columns = cluster_columns
        self.tag_name_columns = tag_name_columns
        return None

    def review_taxonomy(self, data=None) -> pd.DataFrame:
        if data is None:
            data = self.data.copy()
        cluster_pivot = pd.pivot_table(data[self.cluster_columns + ['url']], index=self.cluster_columns, aggfunc=pd.Series.nunique)
        cluster_pivot = cluster_pivot.sort_values(self.cluster_columns, ascending=True)
        cluster_pivot = cluster_pivot.reset_index()
        cluster_pivot = cluster_pivot.drop(columns='url')
        self.cluster_pivot = cluster_pivot
        return cluster_pivot

    def review_taxonomy_subset(self, level=1, cluster_ids=[1]) -> pd.DataFrame:
        data = self.data.copy()
        subset = data[data[f'topic_level_{level}_cluster_id'].isin(cluster_ids)]
        if subset.empty:
            print(f"No data found for level {level} with cluster IDs {cluster_ids}.")
            return pd.DataFrame()
        return self.review_taxonomy(subset)
    
    def view_urls_in_cluster(self, level=1, cluster_id=1) -> pd.DataFrame:
        data = self.data.copy()
        urls = data[data[f'topic_level_{level}_cluster_id'] == cluster_id]['url']
        if urls.empty:
            print(f"No URLs found for level {level} with cluster ID {cluster_id}.")
            return pd.DataFrame()
        
        return urls.reset_index(drop=True)
    
    def manually_rename_cluster(self, level: int, cluster_id: int, new_name: str) -> None:
        column_name = f'topic_level_{level}'
        self.data.loc[self.data[f'topic_level_{level}_cluster_id'] == cluster_id, column_name] = new_name
        return None
    
    def convert_to_tag_name(self, text):
        text = text.split(' ')
        text = '_'.join(text)
        text = text.lower()
        text = text.strip('_')
        return text


