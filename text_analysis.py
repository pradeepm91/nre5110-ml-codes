# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 22:53:59 2020

@author: prade
"""

import bs4 as bs
import urllib.request
import re
import nltk
from gensim.models import Word2Vec

#get input data

scrapped_data = urllib.request.urlopen('https://en.wikipedia.org/wiki/Artificial_intelligence')
article = scrapped_data .read()

parsed_article = bs.BeautifulSoup(article,'lxml')

paragraphs = parsed_article.find_all('p')

article_text = ""

for p in paragraphs:
    article_text += p.text
    
    
# Cleaing the text
processed_article = article_text.lower()
processed_article = re.sub('[^a-zA-Z]', ' ', processed_article )
processed_article = re.sub(r'\s+', ' ', processed_article)
    
# Preparing the dataset
all_sentences = nltk.sent_tokenize(processed_article)

all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

# Removing Stop Words
from nltk.corpus import stopwords
for i in range(len(all_words)):
    all_words[i] = [w for w in all_words[i] if w not in stopwords.words('english')]
    

#create CBOW word2vec model

word2vec = Word2Vec(all_words, min_count=2)

#Create skipgram model
#word2vec = Word2Vec(all_words, min_count=2,sg=1)


vocabulary = word2vec.wv.vocab
print(list(vocabulary))


# %%visualize vectors
v1 = word2vec.wv['artificial']
print(v1)
# %%similar words

sim_words = word2vec.wv.most_similar('intelligence')
print(sim_words)

# %% manipulations
print(word2vec.wv.most_similar(positive=['ai', 'think']))

print(word2vec.wv.most_similar(positive=['ai', 'human'],negative=['machine']))

print(word2vec.wv.most_similar(positive=['ai'] ,negative=['artificial']))

