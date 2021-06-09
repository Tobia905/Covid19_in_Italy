# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 16:19:20 2020

@author: Tobia
"""

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
from nltk.util import ngrams

def text_processing(x, 
                    punct, 
                    stpw, 
                    ng_join = '_',
                    n = None,
                    to_lower=True):
    
    if isinstance(x, pd.core.series.Series):
        
        get_series = True
        
        if to_lower:
            tokens = [word_tokenize(s.lower()) for s in x]
            
        else:
            tokens = [word_tokenize(s) for s in x]
        
        processed_text = []
        for txt in tokens:
            a = [t for t in txt if t not in stpw and t not in punct]
            processed_text.append(a)
            
    elif isinstance(x, str):
        
        get_series = False
        
        if to_lower:
            tokens = word_tokenize(x.lower())
            
        else:
            tokens = word_tokenize(x)
            
        processed_text = [txt for txt in tokens if txt not in punct and txt not in stpw]
        
    if n is not None and get_series == True:
        
        ngram = [list(ngrams(words, n)) for words in processed_text]
        n_ngs = [len(s) for s in ngram]
        
        processed_text = []
        for n, string in enumerate(ngram):
            str_bigs = [ng_join.join(string[t]) for t in range(n_ngs[n])]
            processed_text.append(str_bigs)
    
    elif n is not None and get_series == False:
        
        ngram = list(ngrams(processed_text, n))
        processed_text = [ng_join.join(t) for t in ngram]
            
    return processed_text

def get_counter(corpus, 
                sort = True,
                reverse = True,
                to_df = False):
    
    if isinstance(corpus, pd.core.series.Series):
        
        new_corpus = []
        for txt in corpus:
            for word in txt:
                new_corpus.append(word)
    
    count = collections.Counter(new_corpus)
    
    if sort:
        count = sorted(count.items(), key = lambda x: x[1], reverse = reverse)
        
    if to_df:
        count = pd.DataFrame(count, columns = ['Word', 'Count'])
        
    return count

def get_wordcloud(corpus, 
                  from_freq = True,
                  background_color = 'white', 
                  max_words = 2000,
                  interpolation = 'bilinear',
                  ax = None):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = []
        
    if isinstance(corpus, pd.core.series.Series):
        
        new_corpus = []
        for txt in corpus:
            for word in txt:
                new_corpus.append(word)
        
    wc = WordCloud(background_color=background_color, max_words=max_words)
    
    if from_freq:
        _ = wc.generate_from_frequencies(dict(collections.Counter(new_corpus)))
        
    else:
        _ = wc.generate(corpus)
        
    ax.imshow(wc, interpolation = interpolation, aspect = 'auto')
    _ = ax.axis('off')
    _ = ax.get_xaxis().set_visible(False)
    _ = ax.get_yaxis().set_visible(False)
    
    return fig, ax

def get_vectorizer(corpus, 
                   tfidf = True,
                   fmt = 'df',
                   dense = True):
    
    if tfidf:
        vectorizer = TfidfVectorizer()
        
    else:
        vectorizer = CountVectorizer()
        
    if isinstance(corpus, pd.core.series.Series):
        
        corpus = [' '.join(txt) for txt in corpus]
        
    X = vectorizer.fit_transform(corpus)
    
    if dense:
        X = X.todense()
        
    if fmt == 'df':
        features = vectorizer.get_feature_names()
        word_vec = pd.DataFrame(X.tolist(), columns = features)
        
    elif fmt == 'array':
        word_vec = X.toarray()
        
    return word_vec



