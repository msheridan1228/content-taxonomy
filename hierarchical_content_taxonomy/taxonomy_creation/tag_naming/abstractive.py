#### Old code compilation (from 2021! Do not use this code it won't run) To be broken up and refactored into multiple classes in this folder

# COMMAND ----------

# MAGIC %md # Summarization with one-two words
# MAGIC we want abstractive text summarization

# COMMAND ----------

# Importing requirements
# !pip install --upgraade transformers==4.6.1
# !pip install seq2seq_trainer
# !wget https://raw.githubusercontent.com/huggingface/transformers/master/examples/seq2seq/seq2seq_trainer.py
# !pip install rouge_score
# !pip install pytorch_lightning==0.7.5
# !pip install sentencepiece
# !pip install --upgrade torch==1.8.1
import transformers
from transformers import RobertaTokenizerFast
from transformers import EncoderDecoderModel
import seq2seq_trainer
from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration

# COMMAND ----------

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

# COMMAND ----------

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')
device = torch.device('cpu')
type(tokenizer)

# COMMAND ----------

topic_clusters = docs_df.groupby(["topic_level_4"])['title'].apply(lambda x: '. '.join(x)).reset_index()
topic_clusters.head()

# COMMAND ----------

#text = docs_df['title'][1000]
#text = tag_meta['top_articles'][5]
text = topic_clusters['title'][200]
text = ' '.join(text.split())[:1000]
print(len(text))
print(text)

# COMMAND ----------

text = "summarize:" + text
text

# COMMAND ----------

tokenized_text = tokenizer.encode(text, return_tensors="pt").to(device)

# COMMAND ----------

# summmarize 
summary_ids = model.generate(tokenized_text,
                                    num_beams=4,
                                    no_repeat_ngram_size=2,
                                    min_length=1,
                                    max_length=2,
                                    early_stopping=True)

# COMMAND ----------

output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# COMMAND ----------

print("original text: \n", text)
print ("\n\nSummarized text: \n",output)


# COMMAND ----------


# COMMAND ----------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

#import torch
src_text = [
    text]
#model_name = 'google/pegasus-xsum'
model_name = "facebook/m2m100_418M"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#tokenizer = PegasusTokenizer.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)
#model = PegasusForConditionalGeneration.from_pretrained(model_name, from_tf=True).to(device)
tokenizer.src_lang = "en"
batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt", forced_bos_token_id=tokenizer.get_lang_id("en")).to(device)
translated = model.generate(**batch)
tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
#assert tgt_text[0] == "California's largest electricity provider has turned off power to hundreds of thousands of customers."

# COMMAND ----------

print(tgt_text[0])

# COMMAND ----------

from transformers import pipeline
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
summarizer = pipeline("summarization", model="google/pegasus-xsum", tokenizer=PegasusTokenizer.from_pretrained("google/pegasus-xsum"))

# COMMAND ----------

summarizer(text, min_length=1, max_length=20)

# COMMAND ----------

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import SummarizationPipeline

# COMMAND ----------

model_name = 'lincoln/mbart-mlsum-automatic-summarization'

loaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, from_tf=True)

nlp = SummarizationPipeline(model=loaded_model, tokenizer=loaded_tokenizer)

# COMMAND ----------

nlp(text)
