import re
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_text
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression 
import lightgbm as lgb

from hierarchical_content_taxonomy.taxonomy_creation.text_cleaning import clean_html, first_n_words, embed

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, data = pd.DataFrame(), model=lgb.LGBMClassifier(), num_levels: int = 4):
        self.check_required_columns(data, num_levels)
        self.data = data
        self.num_levels = num_levels
        self.model = model
        
    def check_required_columns(self, data: pd.DataFrame, num_levels: int) -> None:
        required_columns = ['text', 'title', 'url']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Required column '{column}' is missing from the DataFrame.")
        if not all(f'topic_level_{i+1}_cluster_id' in data.columns for i in range(num_levels)):
            raise ValueError(f"DataFrame must contain all topic_level cluster columns up to {num_levels}.")

    def create_pipeline(self) -> Pipeline:
        text_selector = ItemSelector(key = 'text')
        title_selector = ItemSelector(key = 'title')
        url_selector = ItemSelector(key = 'url')

        def clean_html_first_1000(text: str) -> str:
            cleaned = clean_html(text)
            return first_n_words(cleaned, 1000)
        
        clean_html_transformer = CleanHtmlTransformer(cleaner_function=clean_html_first_1000)
        title_pipeline = Pipeline([('title_selector', title_selector),
                            ('title_process', clean_html_transformer)])
        text_pipeline = Pipeline([('text_selector', text_selector),    
                            ('text_process', clean_html_transformer)])

        embedder = EmbedTextTransformer(embed)

        model_pipeline = Pipeline([
            ('union', FeatureUnion(
                transformer_list=[('title', title_pipeline), 
                                ('text', text_pipeline)]
            )),
            ('embedder', embedder),
            ('model', self.model)
        ])
        self.model_pipeline = model_pipeline
        return model_pipeline

    def test_train_split(self, test_size: float = 0.2) -> tuple:
        ## stratify by the last level cluster id
        X = self.data[['text', 'title', 'url']]
        y = self.data[f'topic_level_{self.num_levels}_cluster_id']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self):
        if not hasattr(self, 'model_pipeline'):
            self.create_pipeline()
        self.model_pipeline.fit(self.X_train, self.y_train)
        return self.model_pipeline
    
    def score(self, X=None, y=None):
        if not hasattr(self, 'model_pipeline'):
            raise ValueError("Model pipeline not created. Please run fit() first.")
        if X is None or y is None:
            X = self.X_test
            y = self.y_test
        score = self.model_pipeline.score(X, y)
        return score

    def predict(self, X, y=None):
        if not hasattr(self, 'model_pipeline'):
            raise ValueError("Model pipeline not created. Please run fit() first.")
        predictions = self.model_pipeline.predict(X)
        return self.format_output(predictions)

    # Not complete
    def format_output(self, predictions):
        # Placeholder
        #   id, key, pred
        #   return {"clusterID": str(id), "type":key, "value":pred}
        return predictions

    def get_all_parents(self, subcategory: int, level: int) -> list:
        parent_levels = np.arange(level-1, 0, -1)
        if parent_levels.size == 0:
            return []
        parent_categories = self.data[self.data[f'topic_level_{level}_cluster_id'] == subcategory][[f'topic_level_{parent_level}_cluster_id' for parent_level in parent_levels]]
        if not parent_categories.empty:
            return parent_categories.iloc[0]
        else:
            return None

    # def lookup_parent_category(self, subcategory: int, level: int) -> int:
    #     parent_level = level - 1
    #     if parent_level <= 0:
    #         return None
    #     parent_category = self.data[self.data[f'topic_level_{level}_cluster_id'] == subcategory][f'topic_level_{parent_level}_cluster_id']
    #     if not parent_category.empty:
    #         return parent_category.iloc[0]
    #     else:
    #         return None

    # ## transform to a general function that takes in a dataframe and returns a pivot table
    # def set_tag_level_columns(self) -> None:
    #     cluster_columns = [f'topic_level_{i+1}_cluster_id' for i in range(self.num_levels)]
    #     tag_name_columns = [f'topic_level_{i+1}' for i in range(self.num_levels)]
    #     self.cluster_columns = cluster_columns
    #     self.tag_columns = tag_name_columns
    #     return None

    # def create_cluster_pivot(self) -> pd.DataFrame:
    #     self.set_tag_level_columns()
    #     docs_df = self.data.copy()
    #     if not all(col in docs_df.columns for col in self.cluster_columns):
    #         raise ValueError("DataFrame must contain all cluster columns.")
    #     if 'url' not in docs_df.columns:
    #         raise ValueError("DataFrame must contain 'url' column.")
    #     cluster_pivot = pd.pivot_table(docs_df[self.cluster_columns + ['url']], index=self.cluster_columns, aggfunc=pd.Series.nunique)
    #     cluster_pivot = cluster_pivot.sort_values(self.cluster_columns, ascending=True)
    #     cluster_pivot = cluster_pivot.reset_index()
    #     cluster_pivot = cluster_pivot.drop(columns='url')
    #     self.cluster_pivot = cluster_pivot
    #     return cluster_pivot
    
class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, data_dict):
        return data_dict[self.key]
      
class CleanHtmlTransformer( BaseEstimator, TransformerMixin ):   
  def __init__ (self, cleaner_function):
      self.cleaner_function = cleaner_function
  #Return self nothing else to do here    
  def fit( self, X, y = None ):
      return self 

  #Method that describes what we need this transformer to do
  def transform( self, X, y = None ):
      # Handle both Series and individual strings
      if hasattr(X, 'apply'):
          # If X is a pandas Series, apply the function to each element
          clean_text = X.apply(self.cleaner_function)
      else:
          # If X is a single string, apply directly
          clean_text = self.cleaner_function(X)
      return clean_text

class EmbedTextTransformer( BaseEstimator, TransformerMixin ):   
  def __init__ (self, embedder):
#       self.embedder_ = hub.load(parser)
      self.embedder = embedder
  #Return self nothing else to do here    
  def fit( self, X, y = None ):
      return self 

  #Method that describes what we need this transformer to do
  def transform( self, X, y = None ):
      embedding = self.embedder(X) 
      return embedding

      
