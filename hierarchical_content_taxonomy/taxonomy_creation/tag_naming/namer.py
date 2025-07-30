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

    ## transform to a general function that takes in a dataframe and returns a pivot table
    def set_tag_level_columns(self) -> None:
        cluster_columns = [f'topic_level_{i+1}_cluster_id' for i in range(self.num_levels)]
        tag_name_columns = [f'topic_level_{i+1}' for i in range(self.num_levels)]
        self.cluster_columns = cluster_columns
        self.tag_name_columns = tag_name_columns
        return None
    
    def check_required_columns(self):
        for column in self.required_columns:
            if column not in self.data.columns:
                raise ValueError(f"Required column '{column}' is missing from the DataFrame.")

    @abstractmethod
    def generate_tag_names(self):
        pass
    
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


