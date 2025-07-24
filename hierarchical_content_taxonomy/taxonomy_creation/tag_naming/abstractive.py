#### Old code compilation from 2021. This code is likely not functional as is. WIP to be turned into an extension of the TagNamer base class
import transformers
import numpy as np
import pandas as pd
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, SummarizationPipeline
from hierarchical_content_taxonomy.taxonomy_creation.tag_naming.namer import TagNamer

##inherit from TagNamer base class
##lowkey not impressed with the abstractive stuff. results are not great rn
class AbstractiveTagNamer(TagNamer):

    def __init__(self, data, num_levels, abstractive_model = "facebook/bart-large-cnn"):
        super().__init__(data, num_levels)
        self.model_name = abstractive_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer)

    def generate_tag_names(self):
        self.generate_lowest_level_names()
        self.generate_parent_level_names()
        return self.data

    def generate_lowest_level_names(self):
        data = self.data.copy()
        cluster_column_name = f'topic_level_{self.num_levels}_cluster_id'
        lowest_level_titles = data.groupby([cluster_column_name])['title'].apply(lambda x: '. '.join(x)).reset_index()
        lowest_level_titles[f'topic_level_{self.num_levels}'] = lowest_level_titles.apply(lambda row: self.convert_to_tag_name(self.summarize_text(row['title'], min_length=1, max_length=6)), axis=1)
        name_data = pd.merge(data, lowest_level_titles[[cluster_column_name, f'topic_level_{self.num_levels}']], on=cluster_column_name, how='left')
        self.data = name_data
        return self.data

    def generate_parent_level_names(self):
        data = self.data.copy()
        for level in range(self.num_levels - 1, 0, -1):
            cluster_column_name = f'topic_level_{level}_cluster_id'
            child_column_name = f'topic_level_{level + 1}_cluster_id'
            child_cluster_names = data.groupby([cluster_column_name])[child_column_name].apply(lambda x: '. '.join(x)).reset_index()
            parent_titles = data.apply(lambda row: self.convert_to_tag_name(self.summarize_text(row[child_column_name], min_length=1, max_length=6)), axis=1)
            data = pd.merge(data, parent_titles[[cluster_column_name, f'topic_level_{level}']], on=cluster_column_name, how='left')
        self.data = data
        return self.data

    def summarize_text(self, text, max_new_tokens=15, min_length=1):
        text = text.replace('_', ' ')
        text = ' '.join(text.split())[:1000]
        summary_dict = self.summarizer(text, max_new_tokens=max_new_tokens, min_length=min_length)
        return summary_dict[0]['summary_text']

