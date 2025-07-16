from hierarchical_content_taxonomy.hierarchical_taxonomy import HierarchicalTaxonomy

urls = ['https://example.com/wp-json/wp/v2/posts']
taxonomy = HierarchicalTaxonomy(urls)
taxonomy.pull_data()
taxonomy.generate_cluster_plots()
taxonomy.create_clusters([0.5, 1.0, 1.5])
data = taxonomy.get_data()
print(data.head())
