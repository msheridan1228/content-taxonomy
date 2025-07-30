# Let's test with a simpler approach first - avoiding the heavy model loading
import pandas as pd
from hierarchical_content_taxonomy.taxonomy_creation.tag_naming.namer import TagNamer

class SimpleTagNamer(TagNamer):
    """A simplified version that uses keyword extraction instead of heavy LLMs"""
    
    def __init__(self, data: pd.DataFrame, num_levels: int):
        super().__init__(data, num_levels)
        self.existing_tags = set()
    
    def generate_tag_names(self) -> pd.DataFrame:
        """Generate tag names for all hierarchical levels."""
        self.generate_lowest_level_names()
        self.generate_parent_level_names()
        return self.data
    
    def generate_lowest_level_names(self) -> pd.DataFrame:
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
    
    def _generate_tag_from_titles(self, titles):
        """Generate a tag using simple keyword extraction"""
        import re
        from collections import Counter
        
        # Combine all titles
        text = " ".join(titles).lower()
        
        # Extract words
        words = re.findall(r'\b[a-z]{3,}\b', text)
        
        # Remove common words
        stop_words = {'the', 'and', 'with', 'for', 'easy', 'simple', 'best', 'how', 'recipe', 'make'}
        words = [w for w in words if w not in stop_words]
        
        # Get most common words
        if words:
            word_counts = Counter(words)
            # Take top words
            top_words = [word for word, count in word_counts.most_common(3)]
            tag = "_".join(top_words[:2])  # Use top 2 words
            
            # Ensure uniqueness
            original_tag = tag
            counter = 3
            while tag in self.existing_tags:
                if counter > 6:  # Limit to avoid infinite loop
                    tag = f"{tag}_{counter}"
                    break
                top_words = [word for word, count in word_counts.most_common(counter)]
                tag = "_".join(top_words[:counter])
                counter += 1
            
            self.existing_tags.add(tag)
            return tag
        else:
            return "unknown_topic"
    
    def _generate_parent_tag_from_children(self, child_tags):
        """Generate parent tag from child tags"""
        import re
        from collections import Counter
        
        # Extract words from all child tags
        all_words = []
        for tag in child_tags:
            words = re.split(r'[_\s-]', tag.lower())
            all_words.extend([w for w in words if len(w) > 2])
        
        if all_words:
            word_counts = Counter(all_words)
            # Get most common word
            top_word = word_counts.most_common(1)[0][0]
            
            # Ensure uniqueness
            tag = top_word
            counter = 3
            while tag in self.existing_tags:
                if counter > 6:  # Limit to avoid infinite loop
                    tag = f"{tag}_{counter}"
                    break
                top_words = [word for word, count in word_counts.most_common(counter)]
                tag = "_".join(top_words[:counter])
                counter += 1
            
            self.existing_tags.add(tag)
            return tag
        else:
            return "general_topic"