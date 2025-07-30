#### TF-IDF based tag naming for hierarchical content taxonomy
import nltk
import re
import numpy as np
import pandas as pd
from nltk.stem.porter import *
from collections import Counter 
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from hierarchical_content_taxonomy.taxonomy_creation.tag_naming.namer import TagNamer

# Download required NLTK data
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

class TfidfTagNamer(TagNamer):
    
    def __init__(self, data, num_levels, n_top_words=4):
        super().__init__(data, num_levels)
        self.n_top_words = n_top_words
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.insignificant_words = ['know', 'need', 'new', 'com', 'thing', 'best', 'do', 'I', 'is', 'cause', 'effect', 'vs', 'way', 'come', 'report', 'say', 'amp']
        self.insignificant_pos_tags = ['CD', 'DT', 'CC', 'PRP$', 'EX', 'PRP', 'TO', 'WDT', 'WP', 'WP$', 'WRB']

    def generate_tag_names(self):
        self.generate_lowest_level_names()
        self.generate_parent_level_names()
        return self.data

    def generate_lowest_level_names(self):
        data = self.data.copy()
        cluster_column_name = f'topic_level_{self.num_levels}_cluster_id'
        
        # Create one representative document per cluster (concatenated titles)
        cluster_titles = data.groupby([cluster_column_name])['title'].apply(
            lambda x: '. '.join(x)  # Join all titles in cluster with period separator
        ).reset_index()
        
        # Generate seed words for ALL clusters at once using TF-IDF comparison
        all_cluster_documents = cluster_titles['title'].tolist()
        all_seed_words = generate_seed_words_from_tfidf(all_cluster_documents, self.n_top_words)
        
        # Assign seed words back to each cluster
        cluster_titles['seed_words'] = all_seed_words
        cluster_titles[f'topic_level_{self.num_levels}'] = cluster_titles['seed_words'].apply(
            lambda x: self._create_cluster_name_from_seeds([], x)
        )
        
        data = pd.merge(data, cluster_titles[[cluster_column_name, f'topic_level_{self.num_levels}']], 
                       on=cluster_column_name, how='left')
        self.data = data
        return self.data
    
    def generate_parent_level_names(self):
        data = self.data.copy()
        for level in np.arange(self.num_levels - 1, 0, -1):
            cluster_column_name = f'topic_level_{level}_cluster_id'
            child_column_name = f'topic_level_{level + 1}'
            
            # Create one representative document per parent cluster (concatenated child tag names)
            parent_clusters = data.groupby([cluster_column_name])[child_column_name].apply(
                lambda x: ' '.join(x.unique())  # Join unique child tag names
            ).reset_index()
            
            # Generate seed words for ALL parent clusters at once using TF-IDF comparison
            all_parent_documents = parent_clusters[child_column_name].tolist()
            all_seed_words = generate_seed_words_from_tfidf(all_parent_documents, self.n_top_words)
            
            # Assign seed words back to each parent cluster
            parent_clusters['seed_words'] = all_seed_words
            parent_clusters[f'topic_level_{level}'] = parent_clusters['seed_words'].apply(
                lambda x: self._create_cluster_name_from_seeds([], x)
            )
            
            data = pd.merge(data, parent_clusters[[cluster_column_name, f'topic_level_{level}']], 
                          on=cluster_column_name, how='left') 
        self.data = data
        return self.data

    def _create_cluster_name_from_seeds(self, existing_names, seed_words):
        """Create a valid, unique cluster name from seed words"""
        if not seed_words:
            return 'unnamed'

        clean_name = self._clean_title_word_list(seed_words)

        if clean_name not in existing_names and clean_name != '':
            return clean_name
        return 'unnamed'

    def _clean_title_word_list(self, word_list):
        title_text = ' '.join(word_list)
        tokens = self.tokenize(title_text)
        unique_tokens = list(dict.fromkeys(tokens))  # Remove duplicates while preserving order
        tag_name = '_'.join(unique_tokens[:self.n_top_words])
        tag_name = tag_name.replace('__', '_').replace('--', '').strip('_')
        return tag_name 

    def tokenize(self, title_text):            
        title_text = title_text.lower()
        title_words = re.findall(r'\w+', title_text)
        title_nostop = [word for word in title_words if word not in self.stop_words]
        title_nostop = [word for word in title_nostop if len(word) > 1]
        return title_nostop

    def filter_insignificant_pos(self, word_list):
        good = []
        try:
            pos_tags = nltk.pos_tag(word_list)
            for word, tag in pos_tags:
                if tag not in self.insignificant_pos_tags:
                    good.append(word)
        except:
            good = word_list
        return good

    def filter_insignificant_words(self, word_list):
        return [word for word in word_list if word not in self.insignificant_words]

    def filter_num_words(self, word_list):
        return [word for word in word_list if not any(char.isdigit() for char in word)]

    def lemmatize_word_list(self, word_list):
        return [self.lemmatizer.lemmatize(word) for word in word_list]

    def clean_title_text(self, title_document):
        word_list = self.tokenize(title_document)
        word_list = self.filter_insignificant_pos(word_list)
        word_list = self.lemmatize_word_list(word_list)
        word_list = self.filter_insignificant_words(word_list)
        word_list = self.filter_num_words(word_list)
        return ' '.join(word_list) if word_list else ""

def generate_seed_words_from_tfidf(cluster_documents, n_seed_words=5):    
    """
    Generate seed words for each cluster using TF-IDF across all cluster documents.
    
    Args:
        cluster_documents: List of strings, where each string is the representative text for one cluster
        n_seed_words: Number of top words to return for each cluster
    
    Returns:
        List of lists: Each inner list contains the top TF-IDF words for the corresponding cluster
    """
    # Handle pandas Series input
    if hasattr(cluster_documents, 'tolist'):
        cluster_documents = cluster_documents.tolist()
    elif isinstance(cluster_documents, str):
        cluster_documents = [cluster_documents]
    
    # Ensure all strings are properly cleaned
    cluster_documents = [str(doc).replace('_', ' ') for doc in cluster_documents]
    
    # Remove empty strings but keep track of original indices
    valid_docs = []
    valid_indices = []
    for i, doc in enumerate(cluster_documents):
        if doc.strip():
            valid_docs.append(doc)
            valid_indices.append(i)
    
    if not valid_docs:
        return [[] for _ in cluster_documents]
    
    vectorizer = TfidfVectorizer(
        max_features=max(100, n_seed_words * 10),  # Scale features with seed words needed
        stop_words='english'
    )
    # Fit TF-IDF on all cluster documents at once for proper comparison
    tfidf_matrix = vectorizer.fit_transform(valid_docs)
    feature_names = vectorizer.get_feature_names_out()
    
    # Generate seed words for each cluster document
    all_seed_words = []
    valid_doc_idx = 0
    
    for i in range(len(cluster_documents)):
        if i in valid_indices:
            # Get TF-IDF scores for this specific document
            tfidf_scores = tfidf_matrix[valid_doc_idx].toarray().flatten()
            
            # Get top scoring words for this document
            top_indices = np.argsort(tfidf_scores)[::-1]
            
            # Return top n_seed_words with non-zero scores for this cluster
            seed_words = [feature_names[idx] for idx in top_indices[:n_seed_words] if tfidf_scores[idx] > 0]
            all_seed_words.append(seed_words)
            valid_doc_idx += 1
        else:
            # Empty document, return empty list
            all_seed_words.append([])

    return all_seed_words





