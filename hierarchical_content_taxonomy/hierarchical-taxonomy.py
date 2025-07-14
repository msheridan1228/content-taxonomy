import pandas as pd

class HierarchicalTaxonomy:
    def __init__(self, data):
        self.data = None
        self.max_level = None
        self.tag_columns = None

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

    def set_data(self, data):
        if isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError("Data must be a pandas DataFrame.")

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
        

