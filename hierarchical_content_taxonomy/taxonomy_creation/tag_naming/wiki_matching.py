#### Old code compilation (from 2021! Do not use this code it won't run) To be broken up and refactored into multiple classes in this folder

# COMMAND ----------

# MAGIC %md
# MAGIC # **Using Wiki to generate topic names**

# COMMAND ----------

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

# COMMAND ----------

topic_seed_word_lists = topic_clusters['top_words']

# COMMAND ----------

all_candidate_title_lists_topic =[]
for x in tqdm(topic_seed_word_lists):
    all_candidate_title_lists_topic.append(generateCandidateList(x, nlp))

# COMMAND ----------

semanticMatchDF_list_topic = []
for cl, ql in zip(all_candidate_title_lists_topic, topic_seed_word_lists):
    semanticMatchDF_list_topic.append(compute_semantic_matches(cl,ql))

# COMMAND ----------

lexicalMatchDF_list_topic = []
for cl, ql in zip(all_candidate_title_lists_topic, topic_seed_word_lists):
    lexicalMatchDF_list_topic.append(do_fuzzy_matching(cl,ql))

# COMMAND ----------

combinedDF_list_topic = []
for s,l in zip(semanticMatchDF_list_topic, lexicalMatchDF_list_topic):
    combinedDF_list_topic.append(combine_semantic_lexical_dfs(s,l))

# COMMAND ----------

for i in np.arange(0,len(topic_clusters)):
    print(topic_seed_word_lists[i])
    print(combinedDF_list_topic[i].head(10))
    print("\n\n")

# COMMAND ----------

word = "nurse"
try:
        word_set = wn.synsets(word)[0]
        print(word_set.hypernyms())
        hn_list = sorted([lemma.name() for synset in word_set.hypernyms() for lemma in synset.lemmas()])[:5]
        print(hn_list)
        print(type(hn_list))
except: pass

# COMMAND ----------

test_word = ["nurse"]
word_embedding = embed(word)
test_list = ['programs', 'online', 'nursing', 'nurse', 'rn', 'bsn', 'practitioner', 'degree', 'master', 'msn']
test_sentence = [' '.join(test_list)]
print(test_sentence)
sentence_embedding = embed(test_sentence)
print(len(sentence_embedding[0]))
print(len(word_embedding[0]))
messages  = ["nurse", "programs online nursing nurse rn bsn practitioner degree master msn"]

# COMMAND ----------

distance = scipy.spatial.distance.cdist([word_embedding[0]],  [sentence_embedding[0]], "cosine")[0]

# COMMAND ----------

def embedding_similarity(messages_):
  message_embeddings_ = embed(messages_)
  distance = scipy.spatial.distance.cdist([message_embeddings_[0]],  [message_embeddings_[1]], "cosine")[0]
  print("Similarity score =  {}".format(1-distance))

# COMMAND ----------

embedding_similarity(messages)
