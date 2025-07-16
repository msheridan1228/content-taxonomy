#### Old code compilation from 2021. This code is likely not functional as is. WIP to be turned into an extension of the TagNamer base class

# COMMAND ----------

# MAGIC %md
# MAGIC # **Using Wiki to generate topic names**

# COMMAND ----------
import wikipedia
import sklearn
import numpy as np
import pandas as pd
import tensorflow_hub as hub
#test wiki api
wikipedia.search("covid",results =10)

# COMMAND ----------

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
embed = hub.load(module_url)

# COMMAND ----------

#generates a list of candidate titles given a list of seed words
def generateCandidateList(wordList, nlp):
    # get the top 10 article title results from wikipedia
    titleList = []
    for thisWord in wordList:
        neighbors_10 = wikipedia.search(thisWord,results =10)
        titleList.append(neighbors_10)

    titleList=np.array(titleList).ravel()

    #take the titles and use the spacy model to turn parse them into noun chunks, then de-dupe
    noun_chunk_list = []
    for candidate in titleList:
        doc = nlp(str(candidate))
        for chunk in doc.noun_chunks:
            noun_chunk_list.append(chunk.text)

    noun_chunk_list = np.unique(noun_chunk_list).ravel()

    #combine original title list and noun chunk list
    #noun_chunk_and_title_list = np.concatenate((noun_chunk_list,titleList),axis= None) 
    #or actually lets not add in the original titles
    noun_chunk_and_title_list = noun_chunk_list


    #extract all bigrams from noun chunk list and original title list
    ngram_noun_chunk_and_title_list = []
    for thischunk in noun_chunk_and_title_list:
        vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(1,2))
        try:
            vect.fit([thischunk])
            for x in vect.get_feature_names():
                ngram_noun_chunk_and_title_list.append(x)
        except:
            continue


    #take all of the above and create a large candidate list        
    large_candidate_list = np.concatenate((ngram_noun_chunk_and_title_list, noun_chunk_list),axis=0)
    large_candidate_list =[str.lower(x) for x in large_candidate_list]
    large_candidate_list= np.unique(large_candidate_list)
    
    #should we de-dupe more intelligently?
    
    #the papers also have additional filters
    # only use candidates that have a wiki page of their own
    # use RACO similarity as a filtering mechanism this is based on a deprecated wiki API
    
    return large_candidate_list

# MAGIC %md
# MAGIC ## Mapping to wiki search

# COMMAND ----------

def return_top_wiki_match(thisSemanticMatchDF, thisLexicalMatchDF):
    combined_all = pd.merge(thisSemanticMatchDF,thisLexicalMatchDF)
    #scoring is totally arbitrary
    combined_all["score"] = (0.1*combined_all["setMatch"]+ 0.1*combined_all["sortMatch"])*(combined_all["semantic_dist"]*100)
    return combined_all['word', 'score'].sort_values(by = "score", ascending=False)[0]

def embedding_similarity(messages_):
  message_embeddings_ = embed(messages_)
  distance = scipy.spatial.distance.cdist([message_embeddings_[0]],  [message_embeddings_[1]], "cosine")[0]
  print("Similarity score =  {}".format(1-distance))
