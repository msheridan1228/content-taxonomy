import re
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_text
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression 
import lightgbm as lgb

from hierarchical_content_taxonomy.taxonomy_creation.text_cleaning import clean_html_top_1000, embed

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model=lgb.LGBMClassifier(), num_levels: int = 4):
        self.model = model
        self.num_levels = num_levels
        
    def create_pipeline(self) -> Pipeline:
        text_selector = ItemSelector(key = 'text')
        title_selector = ItemSelector(key = 'title')
        url_selector = ItemSelector(key = 'url')

        clean_html = CleanHtmlTransformer(cleaner_function=clean_html_top_1000)
        title_pipeline = Pipeline([('title_selector', title_selector),
                            ('title_process', clean_html)])
        text_pipeline = Pipeline([('text_selector', text_selector),    
                            ('text_process', clean_html)])
        
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

    def fit(self, X, y=None):
        if not hasattr(self, 'model_pipeline'):
            self.create_pipeline()
        self.model_pipeline.fit(X, y)
        return self.model_pipeline
    
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
    
    def lookup_parent_category(self, subcategory: int) -> int:
        #placeholder
        # Implement logic to find the parent category of a subcategory using tag pivot
        return "parent_category"

    def create_tag_pivot(self, data):
        tag_num_pivot = pd.pivot_table(docs_df[['topic_level_1', 'topic_level_1_cluster_id', 'topic_level_2', 'topic_level_2_cluster_id', 'topic_level_3', 'topic_level_3_cluster_id', 'topic_level_4', 'topic_level_4_cluster_id', 'url']], index=['topic_level_1', 'topic_level_1_cluster_id', 'topic_level_2', 'topic_level_2_cluster_id', 'topic_level_3', 'topic_level_3_cluster_id', 'topic_level_4', 'topic_level_4_cluster_id'], aggfunc=pd.Series.nunique)
        tag_num_pivot = tag_num_pivot.sort_values(['topic_level_1_cluster_id', 'topic_level_2_cluster_id','topic_level_3_cluster_id', 'topic_level_4_cluster_id'], ascending=True)
        tag_num_pivot = tag_num_pivot.reset_index()
        tag_num_pivot = tag_num_pivot.drop(columns='url')
        tag_num_pivot.head()
        return None


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, data_dict):
        return data_dict[self.key]
      
class CleanHtmlTransformer( BaseEstimator, TransformerMixin ):   
  def __init__ (self, cleaner_function):
      self.cleaner_function_ = cleaner_function
  #Return self nothing else to do here    
  def fit( self, X, y = None ):
      return self 

  #Method that describes what we need this transformer to do
  def transform( self, X, y = None ):
      clean_text = self.cleaner_function_(X)
      return clean_text

class EmbedTextTransformer( BaseEstimator, TransformerMixin ):   
  def __init__ (self, parser):
#       self.embedder_ = hub.load(parser)
      self.embedder_ = parser
  #Return self nothing else to do here    
  def fit( self, X, y = None ):
      return self 

  #Method that describes what we need this transformer to do
  def transform( self, X, y = None ):
      embedding = self.embedder_(X) 
      return embedding

      
