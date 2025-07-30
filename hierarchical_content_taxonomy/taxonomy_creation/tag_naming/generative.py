"""
Generative Tag Namer using API endpoints
Uses API calls to language models to generate descriptive tag names for content clusters.
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from typing import List, Optional
import re
import warnings
import time
from hierarchical_content_taxonomy.taxonomy_creation.tag_naming.namer import TagNamer

class GenerativeTagNamer(TagNamer):
    """
    Generative tag namer using API endpoints to create descriptive tag names
    for content clusters based on titles and text content.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 num_levels: int,
                 api_endpoint: str = "https://api.openai.com/v1/chat/completions",
                 api_key: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo",
                 max_tokens: int = 20,
                 temperature: float = 0.2,
                 use_model: bool = True):
        """
        Initialize the GenerativeTagNamer with API endpoint.
        
        Args:
            data: DataFrame with content data
            num_levels: Number of hierarchical levels
            api_endpoint: API endpoint URL for the language model
            api_key: API key for authentication (can also be set via OPENAI_API_KEY env var)
            model_name: Name of the model to use (e.g., "gpt-3.5-turbo", "gpt-4")
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            use_model: Whether to use the model or fallback to keyword extraction
        """
        super().__init__(data, num_levels)
        self.api_endpoint = api_endpoint
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_model = use_model and (self.api_key is not None)
        self.existing_tags = set()  # Track previously generated tags
        
        if not self.use_model:
            print("⚠️  No API key found. Using fallback keyword extraction instead.")
        else:
            print(f"✓ API endpoint configured: {model_name}")

    def generate_lowest_level_names(self) -> pd.DataFrame:
        super().generate_lowest_level_names()
        """Generate tag names for the lowest (most specific) level clusters."""
        data = self.data.copy()
        cluster_column_name = f'topic_level_{self.num_levels}_cluster_id'
        
        # Group titles by cluster
        cluster_groups = data.groupby([cluster_column_name])['title'].apply(
            lambda x: list(x)
        ).reset_index()
        
        # Generate tag names for each cluster
        cluster_groups[f'topic_level_{self.num_levels}'] = cluster_groups['title'].apply(
            lambda titles: self._generate_tag_from_titles(titles)
        )
        
        # Merge back with original data
        data = pd.merge(
            data, 
            cluster_groups[[cluster_column_name, f'topic_level_{self.num_levels}']], 
            on=cluster_column_name, 
            how='left'
        )
        
        self.data = data
        return self.data

    def generate_parent_level_names(self) -> pd.DataFrame:
        super().generate_parent_level_names()
        """Generate tag names for parent level clusters."""
        data = self.data.copy()
        
        for level in range(self.num_levels - 1, 0, -1):
            cluster_column_name = f'topic_level_{level}_cluster_id'
            child_column_name = f'topic_level_{level + 1}'
            
            # Group child tag names by parent cluster
            parent_clusters = data.groupby([cluster_column_name])[child_column_name].apply(
                lambda x: list(x.unique())
            ).reset_index()
            
            # Generate parent tag names from child tags
            parent_clusters[f'topic_level_{level}'] = parent_clusters[child_column_name].apply(
                lambda child_tags: self._generate_parent_tag_from_children(child_tags)
            )
            
            # Merge back with original data
            data = pd.merge(
                data, 
                parent_clusters[[cluster_column_name, f'topic_level_{level}']], 
                on=cluster_column_name, 
                how='left'
            )
        
        self.data = data
        return self.data

    def _generate_tag_from_titles(self, titles: List[str], unavailable_tag = None) -> str:
        """Generate a descriptive tag name from a list of content titles."""
        # Limit number of titles to avoid context length issues
        sample_titles = titles[:10] if len(titles) > 10 else titles

        # Create a prompt for tag generation
        titles_text = ". \n".join(sample_titles)
        prompt = self._create_tag_generation_prompt(titles_text, unavailable_tag)
        
        # Generate tag using the model
        generated_tag = self._generate_with_model(prompt)
        # Clean and format the generated tag
        cleaned_tag = self._clean_generated_tag(generated_tag)
        # Check if the tag already exists or is unavailable
        if (cleaned_tag in self.existing_tags) or (cleaned_tag == unavailable_tag):
            warning_string = f"Generated tag {cleaned_tag} already exists in previous tags. Regenerating with higher temperature."
            self.temperature = min(1, self.temperature + 0.1)  # Increase temperature for more diverse generation
            warnings.warn(warning_string)
            return self._generate_tag_from_titles(titles, unavailable_tag=cleaned_tag)
        self.existing_tags.add(cleaned_tag)
        return cleaned_tag

    def _generate_parent_tag_from_children(self, child_tags: List[str], unavailable_tag = None) -> str:
        """Generate a parent tag name from child tag names."""
        # Remove duplicates and format child tags
        unique_child_tags = list(set(child_tags))
        child_tags_text = ", \n".join(unique_child_tags)
        
        # Create prompt for parent tag generation
        prompt = self._create_parent_tag_prompt(child_tags_text, unavailable_tag)

        # Generate tag using the model
        generated_tag = self._generate_with_model(prompt)
        
        # Clean and format the generated tag
        cleaned_tag = self._clean_generated_tag(generated_tag)
        if (cleaned_tag in self.existing_tags) or (cleaned_tag == unavailable_tag):
            warning_string = f"Generated tag {cleaned_tag} already exists in previous tags. Regenerating with higher temperature."
            warnings.warn(warning_string)
            self.temperature = min(1, self.temperature + 0.1)  # Increase temperature for more diverse generation
            return self._generate_parent_tag_from_children(child_tags, unavailable_tag=cleaned_tag)
        self.existing_tags.add(cleaned_tag)
        return cleaned_tag

    def _create_tag_generation_prompt(self, titles_text: str, unavailable_tag: str = None) -> str:
        """Create a prompt for generating tags from content titles."""
        prompt = f"""Based on these content titles, generate a short descriptive topic that encompasses the high level topic (2-4 words):
        For example, if the child titles are: What to feed your pet mouse. How to change a litter box. How often does my dog need to play? the parent tag could be "animal_care" or "pets".
        or if the titles are: 10 new ways to train a regression model. Unsupervised clustering how to. Hot new machine learning techniques. the tag could be "machine learning".


        Titles:

        {titles_text}"""
        if unavailable_tag:
            prompt += f"\n\nNote: The tag name cannot be the same as: {unavailable_tag}."

        prompt+= "\n Topic:"
        return prompt

    def _create_parent_tag_prompt(self, child_tags_text: str, unavailable_tag:str=None) -> str:
        """Create a prompt for generating parent tags from child tags."""
        prompt = f"""Given these related topic tags, generate a broader category name that encompasses the high level topic (2-4 words):
        For example, if the tags are: animal_care, pets, bird_watching, aquariums, the parent tag could be "animals" or "pets".
        or if the child tags are: regression, clustering, machine_learning, deep_learning, AI, the parent tag could be "machine learning".

        Related topics:  

        {child_tags_text}"""
        if unavailable_tag:
            prompt += f"\n\nNote: The tag name cannot be the same as: {unavailable_tag}."

        prompt+= "\n Topic:"
        return prompt

    def _generate_with_model(self, prompt: str) -> str:
        """Generate text using API call."""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            
            response = requests.post(self.api_endpoint, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result['choices'][0]['message']['content'].strip()
                
                # Extract just the tag part (first line or until punctuation)
                tag = generated_text.split('\n')[0].split('.')[0].split(',')[0].strip()
                return tag
            else:
                print(f"API call failed with status {response.status_code}: {response.text}")
                return "error_tag"
                
        except Exception as e:
            print(f"Error generating with API: {e}")
            return "error_tag"


    def _clean_generated_tag(self, tag: str) -> str:
        """Clean and format the generated tag name."""
        if not tag:
            return "unknown_topic"
            
        # Remove unwanted characters and clean up
        tag = re.sub(r'[^\w\s-]', '', tag)
        tag = re.sub(r'\s+', ' ', tag).strip()
        
        # Convert to standard tag format
        tag_words = tag.lower().split()
        
        # Limit to reasonable number of words
        if len(tag_words) > 3:
            tag_words = tag_words[:3]
            
        # Join with underscores
        clean_tag = "_".join(tag_words)
        
        # Ensure tag is not empty
        if not clean_tag:
            return "general_topic"
            
        return clean_tag

    def set_model_parameters(self, max_tokens: int = None, temperature: float = None):
        """Update model generation parameters."""
        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

    def get_model_info(self) -> dict:
        """Get information about the current model."""
        return {
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_endpoint": self.api_endpoint,
            "use_model": self.use_model
        }
