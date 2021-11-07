# Databricks notebook source
# MAGIC %md # Packages
# MAGIC USE RUNTIME 7.3 LTS (NOT ML)

# COMMAND ----------

# MAGIC %run ./utils/package-utils

# COMMAND ----------

# MAGIC %run ./utils/model-training-utils

# COMMAND ----------

# MAGIC %md 
# MAGIC # Load Prepared Data

# COMMAND ----------

base_output_path = 'user/msheridan/documents/content-taxonomy-'
filepath = '/users/msheridan/documents/marthasblog-clusters.pkl'
  
# COMMAND ----------

with open(filepath, 'rb+') as f:
  docs_df = cloudpickle.load(f)
docs_df.head()

# COMMAND ----------

docs_df['backup_only'] = [0 if is_full_text(text_len) else 1 for text_len in docs_df['text_length']]
print(docs_df.shape)
docs_df['backup_only'].value_counts()

# COMMAND ----------

# MAGIC %md # Create Taxonomy Lookup

# COMMAND ----------

tag_num_pivot = pd.pivot_table(docs_df[['topic_level_1', 'topic_level_1_cluster_id', 'topic_level_2', 'topic_level_2_cluster_id', 'topic_level_3', 'topic_level_3_cluster_id', 'topic_level_4', 'topic_level_4_cluster_id', 'url']], index=['topic_level_1', 'topic_level_1_cluster_id', 'topic_level_2', 'topic_level_2_cluster_id', 'topic_level_3', 'topic_level_3_cluster_id', 'topic_level_4', 'topic_level_4_cluster_id'], aggfunc=pd.Series.nunique)
tag_num_pivot = tag_num_pivot.sort_values(['topic_level_1_cluster_id', 'topic_level_2_cluster_id','topic_level_3_cluster_id', 'topic_level_4_cluster_id'], ascending=True)
tag_num_pivot = tag_num_pivot.reset_index()
tag_num_pivot = tag_num_pivot.drop(columns='url')
tag_num_pivot.head()

# COMMAND ----------

tag_num_pivot[tag_num_pivot['topic_level_4']=='releases_and_updates']

# COMMAND ----------

# MAGIC %md # Create Preprocessing Pipeline
# MAGIC Steps:
# MAGIC - clean html
# MAGIC - embed full html and title
# MAGIC - use tfidf for just title and url

# COMMAND ----------

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]

# COMMAND ----------

text_selector = ItemSelector(key = 'text')
title_selector = ItemSelector(key = 'title')
url_selector = ItemSelector(key = 'url')

# COMMAND ----------

# MAGIC %md ##Backup tfidf model on url and title
# MAGIC for when there is no good body text

# COMMAND ----------

title_vectorizer = TfidfVectorizer(min_df=10, max_df=.9, stop_words='english',analyzer='word', sublinear_tf=True, encoding='latin-1').fit(docs_df['title'])
url_vectorizer =  TfidfVectorizer(min_df=10, max_df=.9, stop_words='english',analyzer='word', sublinear_tf=True, encoding='latin-1').fit(docs_df['url'])

# COMMAND ----------

url_tfidf_pipeline = Pipeline([('selector', url_selector),
                            ('process', url_vectorizer)])
title_tfidf_pipeline = Pipeline([('selector', title_selector),
                            ('process', title_vectorizer)])

# COMMAND ----------

backup_pipeline = Pipeline([
    ('union', FeatureUnion(
         transformer_list=[ ('title', title_tfidf_pipeline),
                           ('url', url_tfidf_pipeline)]
    ))
])

# COMMAND ----------

# MAGIC %md ##Full embedding  model on title and text

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

def embed_row(X, clean_text_pipeline, embed, y=None):
    clean_text = clean_text_pipeline.transform(X)
    x = [' '.join(clean_text)]
    x_embedding = embed(x)
    return x_embedding

# COMMAND ----------

# MAGIC %md # Preprocess Data
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

tfidf = backup_pipeline.fit_transform(docs_df)
print(tfidf.shape)

# COMMAND ----------

backup_tfidf_pipeline_filepath = base_output_path + "backup-tfidf-pipeline.pkl"
with open(backup_tfidf_pipeline_filepath, "wb") as f:
  cloudpickle.dump(backup_pipeline, f)
  
with open(backup_tfidf_pipeline_filepath, "rb") as f:
  backup_pipeline = cloudpickle.load(f)

# COMMAND ----------

# MAGIC %md # Train Supervised Models
# MAGIC Predict most granular (level 4) and inherit higher levels.

# COMMAND ----------

tl1_full_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=24690, solver='lbfgs', n_jobs=-1).fit(embeddings, complete_text_data['topic_level_1_cluster_id'])
tl2_full_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=24690, solver='lbfgs', n_jobs=-1).fit(embeddings, complete_text_data['topic_level_2_cluster_id'])
tl3_full_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=24690, solver='lbfgs', n_jobs=-1).fit(embeddings, complete_text_data['topic_level_3_cluster_id'])
tl4_full_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=24690, solver='lbfgs', n_jobs=-1).fit(embeddings, complete_text_data['topic_level_4_cluster_id'])

full_model_dic = {'tl1': tl1_full_model, 'tl2': tl2_full_model, 'tl3': tl3_full_model, 'tl4': tl4_full_model}

# COMMAND ----------

full_model_dic_filepath = base_output_path + "full-model-dictionary.pkl"

with open(full_model_dic_filepath, "wb") as f:
  cloudpickle.dump(full_model_dic, f)
  
with open(full_model_dic_filepath, "rb") as f:
  full_model_dic = cloudpickle.load(f)

# COMMAND ----------

tl1_tfidf_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=24690, solver='lbfgs', n_jobs=-1).fit(tfidf, docs_df['topic_level_1_cluster_id'])
tl2_tfidf_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=24690, solver='lbfgs', n_jobs=-1).fit(tfidf, docs_df['topic_level_2_cluster_id'])
tl3_tfidf_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=24690, solver='lbfgs', n_jobs=-1).fit(tfidf, docs_df['topic_level_3_cluster_id'])
tl4_tfidf_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=24690, solver='lbfgs', n_jobs=-1).fit(tfidf, docs_df['topic_level_4_cluster_id'])

backup_model_dic = {'tl1': tl1_tfidf_model, 'tl2': tl2_tfidf_model, 'tl3': tl3_tfidf_model, 'tl4': tl4_tfidf_model}

# COMMAND ----------

backup_model_dic_filepath = base_output_path + "backup-model-dictionary.pkl"
with open(backup_model_dic_filepath, "wb") as f:
  cloudpickle.dump(backup_model_dic, f)
  
with open(backup_model_dic_filepath, "rb") as f:
  backup_model_dic = cloudpickle.load(f)

# COMMAND ----------

# MAGIC %md # Create Classifier Pipeline
# MAGIC - predict level 4
# MAGIC - inherit higher levels

# COMMAND ----------

class taxonomy_lookup_classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, tag_pivot, model_dic):
        self.tag_pivot_ = tag_pivot
        self.tl1_model_ = model_dic['tl1']
        self.tl2_model_ = model_dic['tl2']
        self.tl3_model_ = model_dic['tl3']
        self.tl4_model_ = model_dic['tl4']

    def fit(self, X, y=None):
        return self
      
    def predict_proba(self, X, n=1, y=None):  #top n predictions returned
      #for now hard code a number to return
      tag_dictionary = []
      # predict independently to get confidence metrics for inherited values
      tl1_prob_preds = self.tl1_model_.predict_proba(X)
      tl2_prob_preds = self.tl2_model_.predict_proba(X)
      tl3_prob_preds = self.tl3_model_.predict_proba(X)
      tl4_prob_preds = self.tl4_model_.predict_proba(X) 
      def get_confidence(model, tag, embedding, prob_preds):
          prob_index = np.where(model.classes_==tag)[0][0]
          prob_value = prob_preds[0][prob_index]
          return prob_value
      def format_output(id, key, pred, confidence, priority):
          return {"clusterID": id, "type":key, "value":pred, "confidence":float(confidence), "priority":int(priority)}
      for top_n in np.arange(n):  #get tl4 pred and confidence
          level_4_id = self.tl4_model_.classes_[np.argpartition(-tl4_prob_preds, top_n)][0][top_n]
          level_4_confidence = get_confidence(self.tl4_model_, level_4_id, X, tl4_prob_preds)
        #lookup higher level tag values based on tl4 pred
          taxonomy_lookup =  self.tag_pivot_[self.tag_pivot_['topic_level_4_cluster_id']==level_4_id].reset_index()
          print('l4 id: ' +  level_4_id)
          level_1_id = taxonomy_lookup['topic_level_1_cluster_id'][0]
          level_2_id = taxonomy_lookup['topic_level_2_cluster_id'][0]
          level_3_id = taxonomy_lookup['topic_level_3_cluster_id'][0]
          level_1_value = taxonomy_lookup['topic_level_1'][0]
          level_2_value = taxonomy_lookup['topic_level_2'][0]
          level_3_value = taxonomy_lookup['topic_level_3'][0]
          level_4_value = taxonomy_lookup['topic_level_4'][0]
          level_1_confidence = get_confidence(self.tl1_model_, level_1_id, X, tl1_prob_preds)
          level_2_confidence = get_confidence(self.tl2_model_, level_2_id, X, tl2_prob_preds)
          level_3_confidence = get_confidence(self.tl3_model_, level_3_id, X, tl3_prob_preds)
          # potentially output additional "top_k":1,2,3, field
          #dl will reorder anyway
          priority = top_n+1
          tag_dictionary.extend([format_output(level_1_id, "topic_level_1", level_1_value, level_1_confidence, priority), format_output(level_2_id, "topic_level_2", level_2_value, level_2_confidence, priority), format_output(level_3_id, "topic_level_3", level_3_value, level_3_confidence, priority), format_output(level_4_id, "topic_level_4", level_4_value, level_4_confidence, priority)])
      return tag_dictionary

# COMMAND ----------

full_classifier = taxonomy_lookup_classifier(tag_num_pivot, full_model_dic)
backup_classifier = taxonomy_lookup_classifier(tag_num_pivot, backup_model_dic)

# COMMAND ----------

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
    if is_full_text(len_text): ##should update this condition based on performance
      x = [' '.join(clean_text)]
      x_embedding = embed(x)
      y_pred = self.full_model.predict_proba(x_embedding)
    else:
      X['url'] = [X['url']]
      X['title'] = [X['title']]
      x_tfidf = self.backup_text_pipeline.transform(X)
      y_pred = self.backup_model.predict_proba(x_tfidf)
    return y_pred

# COMMAND ----------

model = FinalModel(full_text_pipeline, backup_pipeline, full_classifier, backup_classifier)

# COMMAND ----------

test_pay = pd.DataFrame([
        {
        "sourceUid":"src_1NT28RdkYcgBjTRFLTgYcWcrH1n",
        "identifier": "0011bdbc-298a-4b68-be11-0f5c58712bbe",
        "title": "Yoga for stress",
        "url": "https://healthline.com/yoga-for-stress/",
        "text": "<p> Yoga is known for its ability to ease stress and promote relaxation. </p>"
    },
{
        "sourceUid":"src_1NT28RdkYcgBjTRFLTgYcWcrH1n",
        "identifier": "0011bdbc-298a-4b68-be11-0f5c58712bbe",
        "title": "Yoga for stress",
        "url": "https://healthline.com/yoga-for-stress/",
        "text": "<p> Yoga is known for its ability to ease stress and promote relaxation. In fact, multiple studies have shown that it can decrease the secretion of cortisol, the primary stress hormone (2Trusted Source, 3Trusted Source). One study demonstrated the powerful effect of yoga on stress by following 24 women who perceived themselves as emotionally distressed. After a three-month yoga program, the women had significantly lower levels of cortisol. They also had lower levels of stress, anxiety, fatigue and depression (4Trusted Source). Another study of 131 people had similar results, showing that 10 weeks of yoga helped reduce stress and anxiety. </p>"
    }])

# COMMAND ----------

for index, row in test_pay.iterrows():
  print(index)
  print("title: " + row.title)
  prediction = model.predict_proba(row, embed)
  print("prediction: " + prediction[3]['value'])

# COMMAND ----------

for index, row in docs_df[docs_df['backup_only']==1].sample(10).iterrows():
  print(index)
  print("title: " + row.cleaned_title)
  print(row.backup_only)
  print("topic: " + row['topic_level_4'])
  print(type(row['text']))
  print(row['text'])
  prediction = model.predict_proba(row, embed)
  print("prediction: " + prediction[3]['value'])

# COMMAND ----------

hierarchical_model_filepath = base_output_path + "hierarchical-model"

with open(hierarchical_model_filepath, "wb") as f:
  cloudpickle.dump(model, f)
with open(hierarchical_model_filepath, "rb") as f:
  model = cloudpickle.load(f)
  
