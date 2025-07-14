# COMMAND ----------

import re
import json
import cloudpickle
import pandas as pd
import requests
import numpy as np
import tensorflow as tf
import tensorflow_text
import bs4
from bs4 import BeautifulSoup
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression 
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer
from sklearn.base import BaseEstimator, TransformerMixin



class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
        
    def fit(self, x, y=None):
        return self
    
    def transform(self, data_dict):
        return data_dict[self.key]
      
text_selector = ItemSelector(key = 'text')
title_selector = ItemSelector(key = 'title')
url_selector = ItemSelector(key = 'url')

# COMMAND ----------

# MAGIC %md
# MAGIC # Load Prepared Data

# COMMAND ----------

filepath = '/dbfs/FileStore/datascience/global-tagging-model/latest.pkl'
with open(filepath, 'rb+') as f:
  docs_df = cloudpickle.load(f)   #works with pandas = 1.4.1
docs_df.head()

# COMMAND ----------

docs_df['topic_level_4'].nunique()

# COMMAND ----------

docs_df['topic_level_4_cluster_id'].nunique()

# COMMAND ----------

docs_df['backup_only'] = [0 if is_full_text(text_len) else 1 for text_len in docs_df['text_length']]
print(docs_df.shape)
docs_df['backup_only'].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Taxonomy Lookup

# COMMAND ----------

tag_num_pivot = pd.pivot_table(docs_df[['topic_level_1', 'topic_level_1_cluster_id', 'topic_level_2', 'topic_level_2_cluster_id', 'topic_level_3', 'topic_level_3_cluster_id', 'topic_level_4', 'topic_level_4_cluster_id', 'url']], index=['topic_level_1', 'topic_level_1_cluster_id', 'topic_level_2', 'topic_level_2_cluster_id', 'topic_level_3', 'topic_level_3_cluster_id', 'topic_level_4', 'topic_level_4_cluster_id'], aggfunc=pd.Series.nunique)
tag_num_pivot = tag_num_pivot.sort_values(['topic_level_1_cluster_id', 'topic_level_2_cluster_id','topic_level_3_cluster_id', 'topic_level_4_cluster_id'], ascending=True)
tag_num_pivot = tag_num_pivot.reset_index()
tag_num_pivot = tag_num_pivot.drop(columns='url')
tag_num_pivot.head()

# COMMAND ----------

if environment  == 'development':
  docs_df = docs_df.groupby('topic_level_4_cluster_id', group_keys=False).apply(lambda x: x.sample(20, replace=True))
  print("sampled", len(docs_df))

# COMMAND ----------

# MAGIC %md
# MAGIC # Create Preprocessing Pipeline
# MAGIC Steps:
# MAGIC - clean html
# MAGIC - embed full html and title
# MAGIC - use tfidf for just title and url

# COMMAND ----------

# MAGIC %md
# MAGIC ##Backup tfidf model on url and title
# MAGIC for when there is no good body text

# COMMAND ----------

if manual_updates:

    title_vectorizer = TfidfVectorizer(min_df=10, max_df=.9, stop_words='english',analyzer='word', sublinear_tf=True, encoding='latin-1').fit(docs_df['title'])
    url_vectorizer =  TfidfVectorizer(min_df=10, max_df=.9, stop_words='english',analyzer='word', sublinear_tf=True, encoding='latin-1').fit(docs_df['url'])

    url_tfidf_pipeline = Pipeline([('selector', url_selector),
                                ('process', url_vectorizer)])
    title_tfidf_pipeline = Pipeline([('selector', title_selector),
                                ('process', title_vectorizer)])

    backup_pipeline = Pipeline([
        ('union', FeatureUnion(
            transformer_list=[ ('title', title_tfidf_pipeline),
                            ('url', url_tfidf_pipeline)]
        ))
    ])

# COMMAND ----------

# MAGIC %md
# MAGIC ##Full embedding  model on title and text

# COMMAND ----------

# wraps the above preprocessing helper function into an sklearn pipeline
class clean_html_transformer( BaseEstimator, TransformerMixin ):   

  def __init__ (self, cleaner_function):
      self.cleaner_function_ = cleaner_function
  #Return self nothing else to do here    
  def fit( self, X, y = None ):
      return self 

  #Method that describes what we need this transformer to do
  def transform( self, X, y = None ):
      clean_text = self.cleaner_function_(X)
      return clean_text

# COMMAND ----------

title_full_pipeline = Pipeline([('title_selector', title_selector),
                            ('title_process', clean_html_transformer(cleaner_function = clean_html_top_1000))])
text_full_pipeline = Pipeline([('text_selector', text_selector),
                            ('text_process', clean_html_transformer(cleaner_function = clean_html_top_1000))])

# COMMAND ----------

# wraps the above preprocessing helper function into an sklearn pipeline
class embed_html_transformer( BaseEstimator, TransformerMixin ):   

  def __init__ (self, parser):
#       self.embedder_ = hub.load(parser)
      self.embedder_ = parser
  #Return self nothing else to do here    
  def fit( self, X, y = None ):
      return self 

  #Method that describes what we need this transformer to do
  def transform( self, X, y = None ):
      embedding = self.embedder_(X) 
      return embedding

# COMMAND ----------

embedder = embed_html_transformer(embed)

# COMMAND ----------

full_text_pipeline = Pipeline([
    ('union', FeatureUnion(
         transformer_list=[('title', title_full_pipeline), 
                           ('text', text_full_pipeline)]
    ))
])

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocess Data
# MAGIC create embeddings on full text and tfidf on title and url if those are not present in dataset

# COMMAND ----------

complete_text_data = docs_df[docs_df['backup_only']==0]
print(len(complete_text_data))

# COMMAND ----------

if 'text_embedding' not in complete_text_data.columns:
  embeddings = [embed_row(complete_text_data.iloc[i], full_text_pipeline, embed) for i in np.arange(len(complete_text_data))]
  complete_text_data['text_embedding'] =  [vector.numpy() for vector in embeddings]

embeddings = np.stack(complete_text_data['text_embedding'])
print(embeddings.shape)

# COMMAND ----------

if manual_updates:
    tfidf = backup_pipeline.fit_transform(docs_df)
    print(tfidf.shape)
    backup_tfidf_pipeline_filepath = base_output_path + "backup-tfidf-pipeline.pkl"

    with open(backup_tfidf_pipeline_filepath, "wb") as f:
        cloudpickle.dump(backup_pipeline, f)
    
    try:
        client.log_param(current_run_id, 'backup_tfidf_pipeline_filepath', backup_tfidf_pipeline_filepath)
    except:
        pass

    with open(backup_tfidf_pipeline_filepath, "rb") as f:
        backup_pipeline = cloudpickle.load(f)

# COMMAND ----------

# MAGIC %md
# MAGIC # Train Supervised Models
# MAGIC Predict most granular (level 4) and inherit higher levels.

# COMMAND ----------

class taxonomy_lookup_classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, tag_pivot, topic_level_4_model):
        self.tag_pivot_ = tag_pivot
        self.tl4_model_ = topic_level_4_model


    def fit(self, X, y=None):
        return self
      

    def predict(self, X, y=None):
        tag_prediction = []

        level_4_id = self.tl4_model_.predict(X)
        def format_output(id, key, pred, confidence, priority):
          return {"clusterID": str(id), "type":key, "value":pred}
        taxonomy_lookup =  self.tag_pivot_[self.tag_pivot_['topic_level_4_cluster_id']==level_4_id].reset_index()
        level_1_id = taxonomy_lookup['topic_level_1_cluster_id'][0]
        level_2_id = taxonomy_lookup['topic_level_2_cluster_id'][0]
        level_3_id = taxonomy_lookup['topic_level_3_cluster_id'][0]
        level_1_value = taxonomy_lookup['topic_level_1'][0]
        level_2_value = taxonomy_lookup['topic_level_2'][0]
        level_3_value = taxonomy_lookup['topic_level_3'][0]
        level_4_value = taxonomy_lookup['topic_level_4'][0]
        tag_prediction.extend([format_output(level_1_id, "topic_level_1", level_1_value), format_output(level_2_id, "topic_level_2", level_2_value), format_output(level_3_id, "topic_level_3", level_3_value), format_output(level_4_id, "topic_level_4", level_4_value)])
        return tag_prediction
      

    # def predict_proba(self, X, n=1, y=None):  #top n predictions returned
    #   #for now hard code a number to return
    #   tag_dictionary = []
    #   tl4_prob_preds = self.tl4_model_.predict_proba(X) 
    #   def get_confidence(model, tag, embedding, prob_preds):
    #       prob_index = np.where(model.classes_==tag)[0][0]
    #       prob_value = prob_preds[0][prob_index]
    #       return prob_value
      
    #   def format_output(id, key, pred, confidence, priority):
    #       return {"clusterID": str(id), "type":key, "value":pred, "confidence":float(confidence), "priority":int(priority)}
      
    #   for top_n in np.arange(n):  #get tl4 pred and confidence
    #       level_4_id = self.tl4_model_.classes_[np.argpartition(-tl4_prob_preds, top_n)][0][top_n]
    #       level_4_confidence = get_confidence(self.tl4_model_, level_4_id, X, tl4_prob_preds)
    #     #lookup higher level tag values based on tl4 pred
    #       taxonomy_lookup =  self.tag_pivot_[self.tag_pivot_['topic_level_4_cluster_id']==level_4_id].reset_index()
    #       level_1_id = taxonomy_lookup['topic_level_1_cluster_id'][0]
    #       level_2_id = taxonomy_lookup['topic_level_2_cluster_id'][0]
    #       level_3_id = taxonomy_lookup['topic_level_3_cluster_id'][0]
    #       level_1_value = taxonomy_lookup['topic_level_1'][0]
    #       level_2_value = taxonomy_lookup['topic_level_2'][0]
    #       level_3_value = taxonomy_lookup['topic_level_3'][0]
    #       level_4_value = taxonomy_lookup['topic_level_4'][0]
    #       # potentially output additional "top_k":1,2,3, field
    #       priority = top_n+1
    #       tag_dictionary.extend([format_output(level_1_id, "topic_level_1", level_1_value, level_4_confidence, priority), format_output(level_2_id, "topic_level_2", level_2_value, level_4_confidence, priority), format_output(level_3_id, "topic_level_3", level_3_value, level_4_confidence, priority), format_output(level_4_id, "topic_level_4", level_4_value, level_4_confidence, priority)])

    #   return tag_dictionary
  
class FinalModel(BaseEstimator, TransformerMixin):
  def __init__(self, full_text_pipeline, backup_text_pipeline, full_model, backup_model):
    self.full_text_pipeline = full_text_pipeline
    self.backup_text_pipeline = backup_text_pipeline
    self.full_model = full_model
    self.backup_model = backup_model
    
  def fit(self, X, y=None):
    return self
  
  def predict_proba(self, X, embed, y=None):
    clean_text = self.full_text_pipeline.transform(X)
    len_text = num_words(clean_text[1])
    x = [' '.join(clean_text)]
    x_embedding = embed(x)
    if is_full_text(len_text): ##should update this condition based on performance
      y_pred = self.full_model.predict_proba(x_embedding)
    else:
      X['url'] = [X['url']]
      X['title'] = [X['title']]
      x_tfidf = self.backup_text_pipeline.transform(X)
      y_pred = self.backup_model.predict_proba(x_tfidf)
    return y_pred, x_embedding.numpy()[0].tolist()

# COMMAND ----------

if manual_updates:
    embeddings = [s for s in complete_text_data['text_embedding']]
    embeddings

    docs_df['topic_level_1_cluster_id'] = docs_df['topic_level_1_cluster_id'].apply(str)
    docs_df['topic_level_2_cluster_id'] = docs_df['topic_level_2_cluster_id'].apply(str)
    complete_text_data['topic_level_2_cluster_id'] = complete_text_data['topic_level_2_cluster_id'].apply(str)
    complete_text_data['topic_level_2_cluster_id'] = complete_text_data['topic_level_2_cluster_id'].apply(str)


    tl4_full_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=24690, solver='lbfgs').fit(embeddings, complete_text_data['topic_level_4_cluster_id'])

    full_model_filepath = base_output_path + "full-model.pkl"

    with open(full_model_filepath, "wb") as f:
        cloudpickle.dump(tl4_full_model, f)
    
    try:
        client.log_param(current_run_id, 'full_model_filepath', full_model_filepath)
    except:
        pass

    with open(full_model_filepath, "rb") as f:
        full_model = cloudpickle.load(f)



# COMMAND ----------

if manual_updates:
    tl4_tfidf_model = LogisticRegression(class_weight="balanced", max_iter=10, random_state=24690, solver='lbfgs').fit(tfidf, docs_df['topic_level_4_cluster_id'])

    backup_model_filepath = base_output_path + "backup-model.pkl"

    with open(backup_model_filepath, "wb") as f:
        cloudpickle.dump(tl4_tfidf_model, f)
    
    try:
        client.log_param(current_run_id, 'backup_model_filepath', backup_model_filepath)
    except:
        pass

    with open(backup_model_filepath, "rb") as f:
        backup_model = cloudpickle.load(f)

    

# COMMAND ----------

if manual_updates:

    full_classifier = taxonomy_lookup_classifier(tag_num_pivot, full_model)
    backup_classifier = taxonomy_lookup_classifier(tag_num_pivot, backup_model)

    model = FinalModel(full_text_pipeline, backup_pipeline, full_classifier, backup_classifier)

    hierarchical_model_filepath = base_output_path + "hierarchical-model"

    if environment == 'development':
        hierarchical_model_filepath = hierarchical_model_filepath + "-dev"

    with open(hierarchical_model_filepath, "wb") as f:
        cloudpickle.dump(model, f)

    try:
        client.log_param(current_run_id, 'hierarchical_model_filepath', hierarchical_model_filepath)
    except:
        pass

# COMMAND ----------

hierarchical_model_filepath = '/dbfs/FileStore/datascience/modelData/data-global-content-tagging-development-2025-03-04-hierarchical-model-dev'    

with open(hierarchical_model_filepath, "rb") as f:
    model = cloudpickle.load(f)

output =  [{
    "embedding": [
        -0.02030458115041256, 0.048571400344371796, -0.05271564796566963, -0.05280517414212227, 0.02046523615717888, 0.03052547201514244, -0.06209118291735649, 0.056929297745227814, -0.008686723187565804, 0.006372879724949598, -0.051386892795562744, 0.026010800153017044, 0.05095808207988739, 0.0006469933432526886, -0.007374039385467768, 0.058119699358940125, -0.041616518050432205, -0.056437622755765915, -0.023677341639995575, 0.05703425407409668, 0.010120638646185398, -0.05933675542473793, 0.06342346221208572, -0.048823919147253036, 0.015501363202929497, 0.020971447229385376, 0.06127242371439934, 0.01740274392068386, -0.048677075654268265, -0.03522825241088867, 0.04455449804663658, 0.017491934821009636, -0.062462933361530304, -0.048707157373428345, 0.03611859679222107, -0.016167189925909042, -0.05372960865497589, 0.03466815873980522, 0.03584393486380577, -0.06419572234153748, 0.06346150487661362, -0.01714162528514862, 0.05886837840080261, 0.048819649964571, -0.026335744187235832, 0.03326144069433212, 0.037396110594272614, 0.0632961243391037, 0.059100739657878876, -0.05453741177916527, 0.06450200825929642, -0.05332735925912857, 0.052305422723293304, -0.013956453651189804, 0.013247840106487274, -0.05994901806116104, 0.03234907239675522, -0.061352163553237915, -0.00591053394600749, 0.05386890098452568, -0.06360672414302826, 0.02387346886098385, -0.027622807770967484, -0.020617689937353134, -0.055351484566926956, 0.0633581206202507, -0.04253125563263893, -0.01508225966244936, -0.009036151692271233, 0.028199519962072372, 0.03477303311228752, 0.06066722422838211, 0.05102996155619621, -0.06256290525197983, 0.05441752076148987, 0.00041334284469485283, 0.0327206589281559, -0.033898286521434784, -0.05472274497151375, -0.06453896313905716, -0.06469897925853729, -0.002457910915836692, -0.03756190463900566, -0.058036454021930695, 0.05787096917629242, 0.051911722868680954, 0.05352840572595596, -0.04631409794092178, 0.04731122404336929, 0.05816004425287247, -0.01354397926479578, -0.05977538228034973, 0.015172187238931656, -0.02540225349366665, 0.019871274009346962, 0.05317451059818268, -0.0013666617451235652, 0.059389159083366394, 0.019793789833784103, 0.06309985369443893, 0.05741159990429878, -0.043116990476846695, -0.06293593347072601, 0.055351871997117996, 0.02219797484576702, 0.06474856287240982, -0.056653451174497604, 0.05799565836787224, -0.0398995541036129, 0.05679921805858612, 0.0021947850473225117, -0.02763945795595646, -0.011641987599432468, -0.0584118627011776, 0.03835577890276909, -0.06230273097753525, 0.024023864418268204, 0.06043778359889984, -0.06189873814582825, 0.026811350136995316, 0.0035294939298182726, 0.0236478541046381, -0.03477972000837326, 0.0007883826037868857, -0.04333717003464699, 0.061753373593091965, 0.007357103284448385, 0.0567772202193737, 0.052685387432575226, 0.015504986979067326, 0.0033539547584950924, -0.023387199267745018, 0.044754114001989365, 0.06343702971935272, 0.006708451081067324, 0.05929001420736313, -0.0459897480905056, 0.034511301666498184, 0.03200947120785713, 0.018252652138471603, 0.050905268639326096, 0.04824816808104515, -0.013769521377980709, 0.03239943087100983, -0.008326007053256035, 0.06067575514316559, 0.010346153751015663, 0.001964824041351676, -0.04290499538183212, 0.02796388976275921, -0.04459703713655472, -0.011163509450852871, -0.009672898799180984, -0.06406307220458984, -0.020799225196242332, 0.015325008891522884, 0.06251151114702225, 0.06091797351837158, 0.03437497839331627, -0.04167594388127327, 0.013850432820618153, -0.06396584212779999, -0.05913716182112694, 0.042107291519641876, -0.022371532395482063, 0.02311793528497219, 0.0520888976752758, -0.03690578043460846, -0.0009875578107312322, 0.01509606558829546, -0.05582922324538231, 0.003068347694352269, -0.05846220999956131, -0.06286636739969254, -0.027647895738482475, -0.05291569232940674, 0.06289871037006378, 0.031414344906806946, -0.06347046792507172, -0.05926978960633278, 0.022574298083782196, -0.014029551297426224, -0.007650929503142834, 0.03306864574551582, -0.006725158542394638, 0.03221585601568222, 0.0644645094871521, 0.05616234987974167, 0.06324736773967743, -0.05378349497914314, 0.06156783923506737, -0.03215159848332405, 0.02659856155514717, -0.06283283978700638, -0.015856798738241196, 0.007894638925790787, -0.025773510336875916, 0.05419174209237099, -0.017235824838280678, 0.032987724989652634, -0.002911422634497285, -0.05673770233988762, 0.0551365464925766, -0.06386906653642654, 0.0518498569726944, -0.05274580791592598, 0.05907649174332619, 0.008489228785037994, 0.03600792959332466, -0.0030087854247540236, 0.006858606357127428, 0.05566692352294922, 0.06418720632791519, -0.05502927303314209, -0.04310426488518715, 0.06401482224464417, 0.005894299130886793, 0.005793988239020109, -0.022098034620285034, 0.05418401584029198, -0.023336216807365417, 0.06066429987549782, 0.03987206146121025, -0.0380542054772377, -0.059447117149829865, 0.01417288463562727, 0.04969394579529762, -0.06207425892353058, -0.055411726236343384, -0.06467009335756302, 0.025272876024246216, 0.05732765421271324, -0.02698085643351078, 0.055776242166757584, 0.018168941140174866, 0.006340059917420149, -0.0571897029876709, -0.00941414013504982, -0.03905469924211502, -0.052121471613645554, -0.0599849596619606, 0.025849997997283936, 0.012951264157891273, 0.04483163729310036, 0.05956190079450607, 0.05552578717470169, -0.03811187297105789, -0.05946710333228111, -0.038557905703783035, 0.060424551367759705, 0.007517684251070023, 0.051021505147218704, -0.0005805417313240469, 0.014344016090035439, -0.03242495283484459, -0.061549220234155655, 0.05603223666548729, -0.05798954516649246, -0.022481637075543404, 0.0011209987569600344, -0.04215873405337334, 0.043546099215745926, -0.0554879792034626, 0.05456480011343956, 0.021570462733507156, -0.06200329586863518, -0.021645240485668182, 0.06381827592849731, -0.042398519814014435, 0.0057500386610627174, 0.001308115548454225, 0.06453729420900345, 0.0007872679852880538, -0.03920328617095947, -0.011768675409257412, -0.004033632110804319, 0.0331537239253521, 0.02295696549117565, -0.035109784454107285, -0.0514201857149601, -0.041340943425893784, 0.06274756044149399, 0.0247225109487772, 0.01831757090985775, -0.05072629079222679, -0.016841048374772072, 0.049977388232946396, 0.012891330756247044, -0.010735727846622467, -0.048241011798381805, -0.06249627098441124, -0.047677215188741684, 0.0592537522315979, 0.0645584836602211, 0.04260304197669029, 0.013412545435130596, 0.05248308926820755, 0.05834835395216942, 0.05475269630551338, -0.040025342255830765, 0.001591792912222445, 0.04526888206601143, 0.00039070111233741045, 0.04800218716263771, 0.0316971093416214, -0.04213179647922516, 0.02394498512148857, -0.04702898859977722, 0.06039632856845856, -0.015816809609532356, 0.031660296022892, -0.06368836760520935, 0.056317515671253204, 0.05523810535669327, -0.006857815198600292, 0.03610728681087494, 0.039692703634500504, -0.044837433844804764, 0.06396928429603577, -0.06266900151968002, -0.05187217518687248, -0.06474342197179794, 0.0530550479888916, 0.05620823800563812, 0.038667403161525726, -0.0559408962726593, 0.053112298250198364, 0.052027665078639984, 0.044569071382284164, 0.03964582458138466, -0.04353068396449089, -0.0601235032081604, -0.048841483891010284, 0.01932871900498867, 0.06377390027046204, 0.004523622337728739, -0.052860163152217865, -0.05923976004123688, 0.00949391070753336, 0.04186343401670456, -0.03855498880147934, 0.002526709344238043, 0.045117370784282684, 0.0026546705048531294, -0.06073206663131714, 0.06150924041867256, 0.052648693323135376, 0.05638551712036133, -0.036006439477205276, 0.001810077577829361, -0.017821412533521652, 0.060512274503707886, 0.016610179096460342, -0.06082576513290405, 0.05661588907241821, 0.03102133423089981, -0.053956206887960434, 0.05504738166928291, -0.06319642066955566, 0.06334515661001205, 0.016725031659007072, 0.013099375180900097, 0.028485404327511787, 0.025622718036174774, -0.028087303042411804, -0.06128564849495888, 0.04201946035027504, 0.04310411214828491, 0.03198912739753723, 0.01922062411904335, -0.03152131289243698, -0.005133299622684717, 0.06383760273456573, -0.03738339617848396, -0.01371246948838234, 0.01041277777403593, 0.054980840533971786, -0.028560757637023926, 0.04308237507939339, 0.05784229189157486, 0.05885770171880722, 0.017865458503365517, -0.044879812747240067, 0.06275376677513123, -0.0006433040834963322, 0.05494898557662964, -0.02829027734696865, -0.04843800514936447, 0.04609682410955429, 0.06132107228040695, -0.06101663038134575, -0.019859272986650467, 0.06460680812597275, 0.05977804958820343, -0.014123989269137383, 0.06230415403842926, 0.05282636359333992, -0.04869336634874344, 0.055747926235198975, 0.052296172827482224, -0.047097787261009216, 0.06425120681524277, 0.049019038677215576, -0.03183257207274437, 0.059529244899749756, -0.009601148776710033, -0.001024587661959231, 0.06455910950899124, -0.03605535998940468, 0.06282016634941101, 0.03908190131187439, 0.03977636620402336, -0.006704125087708235, 0.05466200411319733, -0.06323042511940002, 0.06137355417013168, -0.03308799862861633, -0.0034267494920641184, 0.0635886862874031, 0.01099263597279787, -0.012126019224524498, -0.024569371715188026, 0.06135660409927368, -0.02188880369067192, 0.059300489723682404, 0.015790076926350594, -0.00009973242413252592, -0.01729038543999195, -0.05209551006555557, -0.004420020151883364, 0.0545908585190773, 0.036407895386219025, 0.04911456257104874, 0.05011076107621193, -0.0432867631316185, -0.0019269982585683465, 0.0014208270004019141, -0.0002881493419408798, -0.05571651831269264, -0.054009970277547836, -0.011884993873536587, 0.05591871961951256, -0.047005120664834976, -0.035418152809143066, 0.008282918483018875, -0.057013969868421555, 0.05990753695368767, 0.0639219880104065, 0.0632244423031807, 0.005415050778537989, -0.037652406841516495, 0.04980212822556496, -0.06365431100130081, 0.05991491675376892, -0.06467969715595245, -0.055688172578811646, 0.022149071097373962, -0.0030947630293667316, 0.029743917286396027, -0.058949846774339676, 0.06404680013656616, -0.056486599147319794, -0.04703860729932785, -0.05865224450826645, 0.05689366161823273, 0.009944549761712551, -0.05213094502687454, 0.05133611708879471, 0.06378398090600967, 0.04112205654382706, 0.04706137627363205, 0.06472740322351456, -0.04587371647357941, -0.06387399882078171, -0.02058800868690014, 0.052386876195669174, -0.059585511684417725, 0.04859209060668945, 0.01599416881799698, -0.008765296079218388, 0.05026313289999962, 0.06436336040496826, -0.030628947541117668, -0.035300493240356445, 0.05769560486078262, -0.009715177118778229, -0.0579979307949543, 0.034974053502082825, -0.03543351963162422, -0.047928646206855774, -0.05557667091488838, -0.023962533101439476, -0.008617566898465157, -0.060723040252923965, 0.06039681285619736, -0.015329374931752682, -0.007278251461684704, -0.0328512117266655, -0.018954459577798843, -0.03328781947493553, 0.05739253759384155, 0.017236046493053436, -0.012300098314881325, -0.03345458582043648, -0.0014236584538593888, -0.05445530265569687, 0.06400052458047867, 0.05390457436442375, 0.039931826293468475, -0.06302031129598618, -0.03779095783829689, -0.05461383983492851
    ],
    "predictions": [
        {
            "clusterID": "11",
            "confidence": 0.5010039005210488,
            "priority": 1,
            "type": "topic_level_1",
            "value": "health"
        },
        {
            "clusterID": "11.28",
            "confidence": 0.5010039005210488,
            "priority": 1,
            "type": "topic_level_2",
            "value": "neurology"
        },
        {
            "clusterID": "11.28.190",
            "confidence": 0.5010039005210488,
            "priority": 1,
            "type": "topic_level_3",
            "value": "mental_health_disorders"
        },
        {
            "clusterID": "11.28.190.1068",
            "confidence": 0.5010039005210488,
            "priority": 1,
            "type": "topic_level_4",
            "value": "stress"
        }
    ]
}]