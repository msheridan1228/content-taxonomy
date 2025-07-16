#### Old code compilation (from 2021! Do not use this code it won't run) To be broken up and refactored into multiple classes in this folder

# MAGIC %md
# MAGIC # Trying Wordnet

# COMMAND ----------

# !pip install wordnet

# COMMAND ----------

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn

# COMMAND ----------

word = 'dog'
try:
        word_set = wn.synsets(word)[0]
        print(word_set.hypernyms())
        hn_list = sorted([lemma.name() for synset in word_set.hypernyms() for lemma in synset.lemmas()])[:5]
        print(hn_list)
        print(type(hn_list))
except: pass
for synset in wn.synsets(word):
    print("\tLemma: {}".format(synset.name()))
    print("\tDefinition: {}".format(synset.definition()))
    print("\tExample: {}".format(synset.examples()))

# COMMAND ----------

def generate_hypernym_set(seed_word_list):
  hypernym_set = []
  for word in seed_word_list:
      try:
        word_set = wn.synsets(word)[0]
        hn_list = sorted([lemma.name() for synset in word_set.hypernyms() for lemma in synset.lemmas()])[:5]
     #   print(hn_list)
     #   print(type(hn_list))
        if len(hn_list) > 0:
          hypernym_set.extend(hn_list)
      except: pass
  return list(set(hypernym_set))


# COMMAND ----------

all_candidate_hypernym_lists =[]

for i in np.arange(0,len(topic_clusters)):
    print("cluster words:")
    print(topic_seed_word_lists[i])
    for seed_word_list in topic_seed_word_lists:
   #   hypernym_set = generate_hypernym_set(seed_word_list)
      all_candidate_hypernym_lists.append(generate_hypernym_set(seed_word_list))
    print("hypernyms")
    print(all_candidate_hypernym_lists[i])
    print("\n\n")

#for seed_word_list in tqdm(topic_seed_word_lists):


# COMMAND ----------

semanticMatchDF_list_topic = []
for cl, ql in zip(all_candidate_hypernym_lists, topic_seed_word_lists):
    semanticMatchDF_list_topic.append(compute_semantic_matches(cl,ql))

# COMMAND ----------

lexicalMatchDF_list_topic = []
for cl, ql in zip(all_candidate_hypernym_lists, topic_seed_word_lists):
    lexicalMatchDF_list_topic.append(do_fuzzy_matching(cl,ql))

# COMMAND ----------

combinedDF_list_topic = []
for s,l in zip(semanticMatchDF_list_topic, lexicalMatchDF_list_topic):
    combinedDF_list_topic.append(combine_semantic_lexical_dfs(s,l))

# COMMAND ----------

len(combinedDF_list_topic)

# COMMAND ----------

for i in np.arange(0,len(topic_clusters)):
    print(topic_seed_word_lists[i])
    print(combinedDF_list_topic[i].head(10))
    print("\n\n")