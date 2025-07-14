from bs4 import BeautifulSoup
import re
import hub

def clean_html(text):
  if text == None:
    return " "
  text = re.sub('>', '> ',text)
  text = re.sub('<', ' <',text)
  soup = BeautifulSoup(text, "html.parser")
  text = soup.get_text(separator=" ", strip=True)

  # Now extract all the readable text contained in <div data-description="TEXT_HERE"> tags
  for description_tag in soup.find_all("div", attrs={"data-description": True}):
    description_with_html = description_tag["data-description"]
    soup_description = BeautifulSoup(description_with_html, "html.parser")
    text += " " + soup_description.get_text(separator=" ", strip=True)
  for header in soup.find_all("div", attrs={"data-hed": True}):
    header_with_html = header["data-hed"]
    soup_header = BeautifulSoup(header_with_html, "html.parser")
    text += " " + soup_header.get_text(separator=" ", strip=True)
    
  text = text.replace('\n','').replace('\r','').replace('\\', '')
  return text

def num_words(text):
  text = text.split()
  return len(text)

def first_n_words(text, n=1000):
  text = text.split()[:n]
  text = " ".join(text)
  return text

def is_min_text(len_text, min_len=20):
  return len_text > min_len

module_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
embed = hub.load(module_url)
