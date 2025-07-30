"""
Simple Generative Tag Namer - Using DistilBERT with mask filling
Uses DistilBERT masked language model for generating descriptive tag names.
"""

import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFDistilBertForMaskedLM
from typing import List
import re
from collections import Counter
from hierarchical_content_taxonomy.taxonomy_creation.tag_naming.namer import TagNamer

class SimpleGenerativeTagNamer(TagNamer):
    """
    Simple generative tag namer using DistilBERT mask filling.
    """
    
    def __init__(self, data: pd.DataFrame, num_levels: int):
        """
        Initialize with DistilBERT.
        """
        super().__init__(data, num_levels)
        self.existing_tags = set()
        self._load_model()
        
    def _load_model(self):
        """Load DistilBERT model."""
        try:
            print("Loading DistilBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = TFDistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")
            self.use_model = True
            print("✓ DistilBERT model loaded successfully")
            
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("Will use keyword extraction fallback")
            self.use_model = False
            self.model = None
            self.tokenizer = None
            self.tokenizer = None

    def generate_tag_names(self) -> pd.DataFrame:
        """Generate tag names for all levels."""
        self.generate_lowest_level_names()
        self.generate_parent_level_names()
        return self.data

    def generate_lowest_level_names(self) -> pd.DataFrame:
        """Generate tags for lowest level."""
        data = self.data.copy()
        cluster_col = f'topic_level_{self.num_levels}_cluster_id'
        
        cluster_groups = data.groupby([cluster_col])['title'].apply(list).reset_index()
        cluster_groups[f'topic_level_{self.num_levels}'] = cluster_groups['title'].apply(
            self._generate_tag_from_titles
        )
        
        data = pd.merge(data, cluster_groups[[cluster_col, f'topic_level_{self.num_levels}']], 
                       on=cluster_col, how='left')
        
        self.data = data
        return self.data

    def generate_parent_level_names(self) -> pd.DataFrame:
        """Generate parent level tags."""
        data = self.data.copy()
        
        for level in range(self.num_levels - 1, 0, -1):
            cluster_col = f'topic_level_{level}_cluster_id'
            child_col = f'topic_level_{level + 1}'
            
            parent_groups = data.groupby([cluster_col])[child_col].apply(
                lambda x: list(x.unique())
            ).reset_index()
            
            parent_groups[f'topic_level_{level}'] = parent_groups[child_col].apply(
                self._generate_parent_tag
            )
            
            data = pd.merge(data, parent_groups[[cluster_col, f'topic_level_{level}']], 
                           on=cluster_col, how='left')
        
        self.data = data
        return self.data

    def _generate_tag_from_titles(self, titles: List[str]) -> str:
        """Generate a tag from content titles using DistilBERT mask filling."""
        if not self.use_model or self.model is None:
            return self._fallback_tag(titles)
        
        try:
            # Create prompt in the format you specified
            titles_text = "', '".join(titles[:4])  # Limit to 4 titles
            prompt = f"Titles: '{titles_text}'. Topic: [MASK]."
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="tf")
            
            # Get predictions
            logits = self.model(**inputs).logits
            
            # Find mask token index
            mask_token_index = tf.where((inputs.input_ids == self.tokenizer.mask_token_id)[0])
            selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)
            
            # Get predicted token
            predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
            
            # Decode the prediction
            predicted_word = self.tokenizer.decode(predicted_token_id).strip()
            
            # Clean the tag
            tag = self._clean_tag(predicted_word)
            
            # Ensure uniqueness
            if tag in self.existing_tags or not tag:
                tag = self._fallback_tag(titles)
            else:
                self.existing_tags.add(tag)
            
            print(f"Generated '{tag}' from: {titles_text}")
            return tag
            
        except Exception as e:
            print(f"DistilBERT generation failed: {e}")
            return self._fallback_tag(titles)
            
            self.existing_tags.add(tag)
            print(f"Generated tag: '{tag}' from {len(titles)} titles")
            return tag
            
        except Exception as e:
            print(f"Generation failed: {e}")
            return self._fallback_tag(titles)

    def _generate_parent_tag(self, child_tags: List[str]) -> str:
        """Generate parent tag from child tags using DistilBERT mask filling."""
        if not self.use_model or self.model is None:
            return self._fallback_parent_tag(child_tags)
        
        try:
            # Create prompt for parent tag
            tags_text = "', '".join(child_tags[:4])
            prompt = f"Categories: '{tags_text}'. Parent category: [MASK]."
            
            # Same process as title generation
            inputs = self.tokenizer(prompt, return_tensors="tf")
            logits = self.model(**inputs).logits
            mask_token_index = tf.where((inputs.input_ids == self.tokenizer.mask_token_id)[0])
            selected_logits = tf.gather_nd(logits[0], indices=mask_token_index)
            predicted_token_id = tf.math.argmax(selected_logits, axis=-1)
            predicted_word = self.tokenizer.decode(predicted_token_id).strip()
            
            tag = self._clean_tag(predicted_word)
            
            if tag in self.existing_tags or not tag:
                tag = self._fallback_parent_tag(child_tags)
            else:
                self.existing_tags.add(tag)
            
            print(f"Generated parent '{tag}' from: {tags_text}")
            return tag
            
        except Exception as e:
            print(f"Parent DistilBERT generation failed: {e}")
            return self._fallback_parent_tag(child_tags)

    def _clean_tag(self, text: str) -> str:
        """Clean generated text into a proper tag."""
        if not text:
            return ""
        
        # Take first word/phrase, remove punctuation
        tag = text.split('\n')[0].split('.')[0].split(',')[0].strip()
        tag = re.sub(r'[^\w\s]', '', tag)
        tag = re.sub(r'\s+', '_', tag.lower())
        
        # # Limit length
        # if len(tag) > 20:
        #     tag = tag[:20]
            
        return tag

    def _fallback_tag(self, titles: List[str]) -> str:
        """Fallback tag generation using keywords."""
        from collections import Counter
        
        # Extract words from titles
        all_words = []
        for title in titles:
            words = re.findall(r'\b[a-z]{3,}\b', title.lower())
            all_words.extend(words)
        
        # Remove common words
        stop_words = {'the', 'and', 'with', 'for', 'easy', 'best', 'how', 'recipe'}
        filtered = [w for w in all_words if w not in stop_words]
        
        if filtered:
            # Get most common word
            counter = Counter(filtered)
            tag = counter.most_common(1)[0][0]
            
            # Ensure uniqueness
            original = tag
            suffix = 1
            while tag in self.existing_tags:
                tag = f"{original}_{suffix}"
                suffix += 1
                
            self.existing_tags.add(tag)
            return tag
        else:
            return "content"

    def _fallback_parent_tag(self, child_tags: List[str]) -> str:
        """Fallback parent tag generation."""
        from collections import Counter
        
        # Extract words from child tags
        all_words = []
        for tag in child_tags:
            words = tag.split('_')
            all_words.extend(words)
        
        if all_words:
            counter = Counter(all_words)
            parent_tag = counter.most_common(1)[0][0]
            
            # Ensure uniqueness
            original = parent_tag
            suffix = 1
            while parent_tag in self.existing_tags:
                parent_tag = f"{original}_{suffix}"
                suffix += 1
                
            self.existing_tags.add(parent_tag)
            return parent_tag
        else:
            return "general"
