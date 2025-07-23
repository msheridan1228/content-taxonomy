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

from hierarchical_content_taxonomy.text_cleaning import CleanHtmlTransformer, EmbeddingTransformer, TitleTextConcatenator

class HierarchicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, data = pd.DataFrame(), model=lgb.LGBMClassifier(), num_levels: int = 4, embed_module_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"):
        self.check_required_columns(data, num_levels)
        self.data = data
        self.num_levels = num_levels
        self.model = model
        self.clean_text_transformer = CleanHtmlTransformer()
        self.embedding_transformer = EmbeddingTransformer(module_url=embed_module_url)
        self.model_pipeline = Pipeline([
                ('concat', TitleTextConcatenator()),
                ('clean_html', self.clean_text_transformer),
                ('embed', self.embedding_transformer),
                ('model', self.model)
            ])

    def check_required_columns(self, data: pd.DataFrame, num_levels: int) -> None:
        required_columns = ['text', 'title', 'url']
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"Required column '{column}' is missing from the DataFrame.")
        if not all(f'topic_level_{i+1}_cluster_id' in data.columns for i in range(num_levels)):
            raise ValueError(f"DataFrame must contain all topic_level cluster columns up to {num_levels}.")

    def test_train_split(self, test_size: float = 0.2) -> tuple:
        ## stratify by the last level cluster id
        X = self.data[['text', 'title', 'url']]
        y = self.data[f'topic_level_{self.num_levels}_cluster_id']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, stratify=y)
        ##check that we have at least 2 samples of every class in training
        classes_in_training = np.unique(self.y_train)
        class_sizes = [len(np.where(self.y_train == cls)[0]) for cls in classes_in_training]
        if any([class_size < 2 for class_size in class_sizes]):
            raise ValueError("Not enough samples in training set for each class. Please adjust test_size or data.")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def fit(self):
        self.model_pipeline.fit(self.X_train, self.y_train)
        return self.model_pipeline
    
    def score(self, X=None, y=None):
        if not hasattr(self, 'model_pipeline'):
            raise ValueError("Model pipeline not fitted. Please run fit() first.")
        if X is None or y is None:
            X = self.X_train
            y = self.y_train
        score = self.model_pipeline.score(X, y)
        return score

    def predict(self, X, y=None):
        if not hasattr(self, 'model_pipeline'):
            raise ValueError("Model pipeline not fitted. Please run fit() first.")
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

      
