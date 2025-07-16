#### Old code compilation (from 2021! Do not use this code it won't run) To be broken up and refactored into multiple classes in this folder

# COMMAND ----------

#load up a spacy model for grammar parsing and noun chunking
nlp = spacy.load("en_core_web_sm")

# COMMAND ----------

def compute_semantic_matches(this_candidate_list, this_query_list):
    #first, need to turn the seed list and all possible candidates into semantic vectors
    searchString =str(this_query_list).translate(str.maketrans('', '', string.punctuation))
    searchVector = embed([searchString])
    vect_list =np.array(embed(this_candidate_list))

    #then, compute distance of each candidate from the searchstring vector
    distList = []
    for thisVector in vect_list: 
        #euclidian distance
        #thisDist = np.linalg.norm(searchVector-thisVector)
        #cosine distance
        #thisDist = scipy.spatial.distance.cosine(searchVector,thisVector)
        #dot product
        thisDist = np.dot(searchVector,thisVector)[0]
        #print(thisDist)
        distList.append(thisDist)

    semanticDistanceList = pd.DataFrame([this_candidate_list,distList]).T
    semanticDistanceList.columns = ['word', 'semantic_dist']
    semanticDistanceList.sort_values(by = 'semantic_dist', ascending=False, inplace= True)
    
    return semanticDistanceList

# COMMAND ----------

def do_fuzzy_matching(this_candidate_list,this_query_list ):
    searchString =str(this_query_list).translate(str.maketrans('', '', string.punctuation))
    fuzzMatchSetDF = pd.DataFrame(process.extract(searchString, this_candidate_list, limit = len(this_candidate_list), scorer = fuzz.token_set_ratio), columns = ['word', 'setMatch'])
    fuzzMatchSortDF = pd.DataFrame(process.extract(searchString, this_candidate_list, limit = len(this_candidate_list), scorer = fuzz.token_sort_ratio), columns = ['word', 'sortMatch'])
    combined_lex = pd.merge(left =fuzzMatchSetDF, right = fuzzMatchSortDF)
    
    return combined_lex

# should also implement some other metods of lexical similarity
# jaccard similarity, levenshtein distance, q-gram distance, etc.

# COMMAND ----------

def combine_semantic_lexical_dfs(thisSemanticMatchDF, thisLexicalMatchDF):
    combined_all = pd.merge(thisSemanticMatchDF,thisLexicalMatchDF)
    #scoring is totally arbitrary
    combined_all["score"] = (0.1*combined_all["setMatch"]+ 0.1*combined_all["sortMatch"])*(combined_all["semantic_dist"]*100)
    return combined_all.sort_values(by = "score", ascending=False)[0:20]

# COMMAND ----------



# COMMAND ----------

with open('eng-com_web-public_2018_10K/eng-com_web-public_2018_10K-sentences.txt', 'r') as f:
    lines = f.readlines()

# COMMAND ----------

print(lines[0])
all_sentences = [l.split('\t')[1] for l in lines]

# COMMAND ----------

# MAGIC %## Create a search index

# COMMAND ----------


from sklearn.neighbors import KDTree
import numpy as np


class ContextNeighborStorage:
    def __init__(self, sentences, model):
        self.sentences = sentences
        self.model = model

    def process_sentences(self):
        result = self.model(self.sentences)

        self.sentence_ids = []
        self.token_ids = []
        self.all_tokens = []
        all_embeddings = []
        for i, (toks, embs) in enumerate(tqdm(result)):
            for j, (tok, emb) in enumerate(zip(toks, embs)):
                self.sentence_ids.append(i)
                self.token_ids.append(j)
                self.all_tokens.append(tok)
                all_embeddings.append(emb)
        all_embeddings = np.stack(all_embeddings)
        # we normalize embeddings, so that euclidian distance is equivalent to cosine distance
        self.normed_embeddings = (all_embeddings.T / (all_embeddings**2).sum(axis=1) ** 0.5).T

    def build_search_index(self):
        # this takes some time
        self.indexer = KDTree(self.normed_embeddings)

    def query(self, query_sent, query_word, k=10, filter_same_word=False):
        toks, embs = self.model([query_sent])[0]

        found = False
        for tok, emb in zip(toks, embs):
            if tok == query_word:
                found = True
                break
        if not found:
            raise ValueError('The query word {} is not a single token in sentence {}'.format(query_word, toks))
        emb = emb / sum(emb**2)**0.5

        if filter_same_word:
            initial_k = max(k, 100)
        else:
            initial_k = k
        di, idx = self.indexer.query(emb.reshape(1, -1), k=initial_k)
        distances = []
        neighbors = []
        contexts = []
        for i, index in enumerate(idx.ravel()):
            token = self.all_tokens[index]
            if filter_same_word and (query_word in token or token in query_word):
                continue
            distances.append(di.ravel()[i])
            neighbors.append(token)
            contexts.append(self.sentences[self.sentence_ids[index]])
            if len(distances) == k:
                break
        return distances, neighbors, contexts

# COMMAND ----------

storage = ContextNeighborStorage(sentences=all_sentences, model=bert)
storage.process_sentences()
storage.build_search_index()

# COMMAND ----------

# MAGIC %md ## Query homonymous words

# COMMAND ----------

distances, neighbors, contexts = storage.query(
     query_sent='It is an investment bank.', query_word='bank', k=5, filter_same_word=True)