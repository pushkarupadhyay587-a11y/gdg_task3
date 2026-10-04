import html
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL = re.compile(r"https?://|www\.\s+|\s+")
NON_ALNUM = re.compile(r"[^a-z0-9$%\s]")
SPACES = re.compile(r"\s+")


def preprocessor(df):
    df["Title"] = df["Title"].str.lower()
    
    df["Title"] = df['Title'].str.replace(r'\s*\([^)]*\)$', '')
    
    df["Description"] = df["Description"].apply(lambda x : x.replace("\\"," "))
    df['Description'] = df['Description'].str.replace(r'^[^-]+-\s*', '')
    
    
    df["Description"]=df["Description"].apply(remove_html_tags)
    df["Description"]=df["Description"].apply(remove_url)
    df["Description"]=df["Description"].apply(remove_punc)
    
    df["Title"]=df["Title"].apply(remove_html_tags)
    df["Title"]=df["Title"].apply(remove_url)
    df["Title"]=df["Title"].apply(remove_punc)
    stop_words = ENGLISH_STOP_WORDS
    df['Description'] = df['Description'].str.split().apply(
    lambda words: " ".join(
    word for word in words
    if word.lower() not in stop_words
    )
    )
    
    df['Title'] = df['Title'].str.split().apply(
    lambda words: " ".join(
    word for word in words
    if word.lower() not in stop_words
    )
    )   
    df["clean"] = df["Title"] + " " + df["Description"]
    
    
import re
def remove_html_tags(text):
    pattern = re.compile('<.*?>')
    return pattern.sub(r'', text)

def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

import string 
exclude = string.punctuation

def remove_punc(text):
    for char in exclude:
        text = text.replace(char,'')
    return text
        