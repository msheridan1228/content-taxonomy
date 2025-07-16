# Hierarchical Content Taxonomy
This repo is designed to generate a hierarchical topic tagging taxonomy based on website articles.
The goals of this work are to:
* pull data article data dynamically from a website
* clean that data and embed it for agglomerative clustering
* help the end user pick level of granularity of their hierarchical topic tags (WIP)
* automatically generate names of those tags based on the content within them (WIP)
* train a classifier to predict the topics of any newly published content (WIP)

Not all of the following goals are fully implemented as of July 2025, marked as WIP. 

Taxonomy Creation:
* wordpress scraping:
* * Content can be scraped from a Wordpress site with the /posts/ endpoint exposed. User inputs list of urls for scraping. Please be mindful of usage restrictions here.
* * The HTML of the title and body text are cleaned and returned. Text cleaning file contains helper functions there.
* * Old posts from before 2020 are removed - may want to adjust this later

* cluster creation
* * Title + text of content is embedded using the Universal Sentence Encoder
* * The embeddings are fed into an agglomerative clustering method to generate a tree
* * Information about that tree is presented to the user - would like to add more visibility methods here
* * User selects cut off points to create hierarchical taxonomy
* * Cluster IDs are assigned to content
## Example silhouette score plot generated from topic clustering
<img width="1013" height="701" alt="silhouette" src="https://github.com/user-attachments/assets/a1a5d25a-7827-4d4a-9a2f-b52b261d6b45" />
## Example dendrogram plot generated from topic clustering
<img width="814" height="604" alt="dendrogram" src="https://github.com/user-attachments/assets/fce5fa83-ea19-46c3-bcae-f10de329c80e" />


* tag naming
* * Want to implement a generalized base class in this file and extend it with different tag naming methods
* * This code still needs significant refactoring and is not functional today. Leaving it here so you can see where I'm headed.

Taxonomy Classification:
* hierarchical classifier:
* * This file contains code to train a classifier to predict the lowest (most granular) taxonomy topic and return the full topic tree
* * This code still is a work in progress. Need to add taxonomy look up, as well as data splitting, model tuning and evaluation. 

Github copilot was used in refactoring my existing very old and messy code on this topic

