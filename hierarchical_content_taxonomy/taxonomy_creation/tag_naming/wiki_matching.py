#### Wikipedia-based tag naming for hierarchical content taxonomy
from torch import ge
import wikipedia
import sklearn
import numpy as np
import pandas as pd
import tensorflow_text
import tensorflow_hub as hub
import scipy.spatial.distance
import spacy
from hierarchical_content_taxonomy.taxonomy_creation.tag_naming.namer import TagNamer
from hierarchical_content_taxonomy.taxonomy_creation.tag_naming.tfidf import generate_seed_words_from_tfidf
from hierarchical_content_taxonomy.text_cleaning import clean_html
class WikiMatchingTagNamer(TagNamer):
    
    def __init__(self, data, num_levels, nlp_model=None):
        super().__init__(data, num_levels)
        
        # Set default spaCy model if none provided
        if nlp_model is None:
            try:
                self.nlp_model = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: en_core_web_sm not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp_model = None
        else:
            self.nlp_model = nlp_model  # spaCy model for noun chunk extraction

        module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
        self.embed = hub.load(module_url)
        
        # Test wiki api
        test_results = wikipedia.search("cats", results=10)
        print(f"Wikipedia API test successful. Found {len(test_results)} results for 'cats'")

    def generate_lowest_level_names(self):
        super().generate_lowest_level_names()

        data = self.data.copy()
        cluster_column_name = f'topic_level_{self.num_levels}_cluster_id'
        cluster_titles = data.groupby([cluster_column_name])['title'].apply(lambda x: '. '.join(x)).reset_index()
        cluster_titles['seed_words'] = generate_seed_words_from_tfidf(cluster_titles['title'])
        cluster_titles[f'topic_level_{self.num_levels}'] = cluster_titles['seed_words'].apply(
            lambda x: self._generate_candidate_list(x)
        )
        data = pd.merge(data, cluster_titles[[cluster_column_name, f'topic_level_{self.num_levels}']], 
                       on=cluster_column_name, how='left')
        self.data = data
        return self.data

    def generate_parent_level_names(self):
        super().generate_parent_level_names()

        data = self.data.copy()
        for level in np.arange(self.num_levels - 1, 0, -1):
            cluster_column_name = f'topic_level_{level}_cluster_id'
            
            # Create one representative document per parent cluster (concatenated child tag names)
            parent_clusters = data.groupby([cluster_column_name])['title'].apply(
                lambda x: ' '.join(x.unique())
            ).reset_index()
            
            # Generate seed words for ALL parent clusters at once using TF-IDF comparison
            all_parent_documents = parent_clusters['title'].tolist()
            all_seed_words = generate_seed_words_from_tfidf(all_parent_documents, 5)
            
            # Assign seed words back to each parent cluster
            parent_clusters['seed_words'] = all_seed_words
            parent_clusters[f'topic_level_{level}'] = parent_clusters['seed_words'].apply(
                lambda x: self._generate_candidate_list(x)
            )
            data = pd.merge(data, parent_clusters[[cluster_column_name, f'topic_level_{level}']], 
                          on=cluster_column_name, how='left')
        
        self.data = data
        return self.data

    def _generate_candidate_list(self, word_list):
        if not word_list or not self.nlp_model:
            return []
        title_list = []
        for this_word in word_list:
            try:
                neighbors = wikipedia.search(this_word, results=1)
                title_list.extend(neighbors)  # Use extend instead of append
            except:
                continue

        # clean titles 
        title_list = [clean_html(title.lower()) for title in title_list if isinstance(title, str)]
        # remove "disambiguation" from titles
        title_list = [title.replace("disambiguation", "").strip() for title in title_list]

        #take the titles and use the spacy model to turn parse them into noun chunks, then de-dupe
        noun_chunk_list = []
        for candidate in title_list:
            try:
                doc = self.nlp_model(str(candidate))
                for chunk in doc.noun_chunks:
                    noun_chunk_list.append(chunk.text)
            except:
                continue

        #should we de-dupe more intelligently?
        # return most frequent 4 words concatenated
        top_candidates = sorted(noun_chunk_list, key=lambda x: len(x.split()), reverse=True)[:4]
        #return most frequent 3 words concatenated
        tag_name = '_'.join(top_candidates)
        tag_name = self.convert_to_tag_name(tag_name)
        #the papers also have additional filters
        # only use candidates that have a wiki page of their own
        # use RACO similarity as a filtering mechanism this is based on a deprecated wiki API
        return tag_name