#!/usr/bin/env python
# coding: utf-8
## To do - make some functions,varaibles private so that they are not accessible outside this file. Some other clean-up to be done.

import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime
from datetime import date
from datetime import timedelta

from pathlib import Path
import requests
import json
import os
from dotenv import load_dotenv

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from string import punctuation
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import re

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


def vader_sentiment_summarizer(df_column):

    vader_analyzer = SentimentIntensityAnalyzer()

    vader_sentiments = []
    for text in df_column:

        sentiment = vader_analyzer.polarity_scores(text)

        compound = sentiment["compound"]
        pos = sentiment["pos"]
        neu = sentiment["neu"]
        neg = sentiment["neg"]

        vader_sentiments.append({
            "Compound": compound,
            "Positive": pos,
            "Neutral": neu,
            "Negative": neg,
            "Text": text})

    vader_sentiments_df = pd.DataFrame.from_dict(vader_sentiments)

    return vader_sentiments_df

def get_vader_sentiment(input_df_column, output_csv_path):
    vader_df = vader_sentiment_summarizer(input_df_column)
    vader_df.to_csv(output_csv_path, header=vader_df.columns)
    print ("file saved: " + output_csv_path)

# # Sentiment based on TextBlob with LEMMATIZER

def text_blob_lemmatizer(df_column):

    wnl = WordNetLemmatizer()

    # Set stop words
    stop = stopwords.words('english')
    stop.append("u")
    stop.append("it'")
    stop.append("'s")
    stop.append("n't")
    stop.append("…")
    stop.append("\`")
    stop.append("``")
    stop.append("char")
    stop.append("''")
    stop.append("'")
    stop.append("/")
    stop.append("r")
    stop.append(" ")
    stop.append("chars")
    stop.append("...")
    stop = set(stop)

    new_column = df_column.apply(lambda x: " ".join(x.lower() for x in x.split()))
    new_column = new_column.str.replace("[^a-zA-Z0-9]"," ")
    new_column = new_column.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    new_column = new_column.apply(lambda x: " ".join([wnl.lemmatize(word) for word in x.split()]))

    return new_column

# Create the textblob sentiment function

def textblob_sentiment_summarizer(df_column):
    polarity = TextBlob(df_column).sentiment.polarity
    return polarity


def get_textblob_lemmatized_sentiment(input_df_column, output_csv_path):
    textblob_lemmatized = pd.DataFrame(text_blob_lemmatizer(input_df_column))
    textblob_lemmatized_polarity = textblob_lemmatized.full_text.apply(textblob_sentiment_summarizer)
    textblob_lemmatized_polarity.to_csv(output_csv_path)
    print ("file saved: " + output_csv_path)


# # Sentiment based on TextBlob with STEMMER

# In[11]:


""" Create the Tokenizer function with STEMMER """

def text_blob_stemmer(df_column):

    st = PorterStemmer()

    # Set stop words
    stop = stopwords.words('english')
    stop.append("u")
    stop.append("it'")
    stop.append("'s")
    stop.append("n't")
    stop.append("…")
    stop.append("\`")
    stop.append("``")
    stop.append("char")
    stop.append("''")
    stop.append("'")
    stop.append("/")
    stop.append("r")
    stop.append(" ")
    stop.append("chars")
    stop.append("...")
    stop = set(stop)

    new_column = df_column.apply(lambda x: " ".join(x.lower() for x in x.split()))
    new_column = new_column.str.replace("[^a-zA-Z0-9]"," ")
    new_column = new_column.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    new_column = new_column.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

    return new_column


def get_textblob_stemmer_sentiment(input_df_column, output_csv_path):
    textblob_stemmed = pd.DataFrame(text_blob_stemmer(input_df_column))
    textblob_stemmed_polarity = textblob_stemmed.full_text.apply(textblob_sentiment_summarizer)
    textblob_stemmed_polarity.to_csv(output_csv_path)
    print ("file saved: " + output_csv_path)


# # Calculating the TF-IDF Weights of Tesla

def create_tesla_news_df(input_df_column):
    import matplotlib as mpl
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer

    plt.style.use("seaborn-whitegrid")
    mpl.rcParams["figure.figsize"] = [20.0, 10.0]

    # Set stop words

    sw = set(stopwords.words('english'))
    sw.update(("u", "it'", "'s", "n't", "…", "\`", "``", "char", "chars", "''","'", "r", " ", "...", "000"))

    # Calculating TF-IDF for the working corpus.

    vectorizer = TfidfVectorizer(stop_words=sw)
    X = vectorizer.fit_transform(input_df_column)

    # Creating a DataFrame Representation of the TF-IDF results

    tesla_news_df = pd.DataFrame(
        list(zip(vectorizer.get_feature_names(), np.ravel(X.sum(axis=0)))),
        columns=["Word", "Frequency"])

    # Order the DataFrame by word frequency in descending order

    tesla_news_df = tesla_news_df.sort_values(
        by=["Frequency"], ascending=False)

    return tesla_news_df

# Top words will have frequency between 10 & 30 (rule of thumb)
def create_words_cloud_diagrams(input_df_column):
    tesla_news_df = create_tesla_news_df(input_df_column)
    tesla_top_words = tesla_news_df[
        (tesla_news_df["Frequency"] >= 10) & (
            tesla_news_df["Frequency"] <= 30)]

    # Create a string list of terms to generate the word cloud
    terms_list = str(tesla_top_words["Word"].tolist())

    # Create the word cloud
    wordcloud2 = WordCloud(colormap="RdYlBu",
                           background_color="white"
                          ).generate(terms_list)
    plt.imshow(wordcloud2)
    plt.axis("off")
    fontdict = {"fontsize": 20, "fontweight": "bold"}
    plt.title("Top Tesla News Word Cloud", fontdict=fontdict)
    plt.show()

# # Creating the Tesla Word Cloud

# Create a string list of ALL words to generate the word cloud
    all_tesla_words = str(tesla_news_df["Word"].tolist())

# Create the word cloud
    wordcloud1 = WordCloud(colormap="RdYlBu",
                       background_color="white"
                      ).generate(all_tesla_words)
    plt.imshow(wordcloud1)
    plt.axis("off")
    fontdict = {"fontsize": 20, "fontweight": "bold"}
    plt.title("Tesla News Word Cloud", fontdict=fontdict)
    plt.show()
#

# https://towardsdatascience.com/create-word-cloud-into-any-shape-you-want-using-python-d0b88834bc32


    from PIL import Image
    from wordcloud import ImageColorGenerator

    tesla_icon = "./images/tesla_shell.png"
    char_mask = np.array(Image.open(tesla_icon))
    # image_colors = ImageColorGenerator(char_mask)

    wc = WordCloud(colormap="copper",
                   background_color="white",
                   width=100, height=100,
                   mask=char_mask,
                   contour_width=5, contour_color="grey",
                   min_font_size=15,
                   random_state=5
                  ).generate(all_tesla_words)

    # plt.figure(facecolor="white", edgecolor="black")
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    fontdict = {"fontsize": 20, "fontweight": "bold"}
    plt.title("Tesla News Word Cloud", fontdict=fontdict)
    plt.tight_layout(pad=0)
    plt.show

#
# # # Create Visualizations using NLPlot
#
# # https://www.kaggle.com/takanobu0210/twitter-sentiment-eda-using-nlplot
#
# # In[24]:
#
#
# import nlplot
# from plotly.subplots import make_subplots
# import plotly.express as px
#
# pd.set_option('display.max_columns', 300)
# pd.set_option('display.max_rows', 300)
# pd.options.display.float_format = '{:.3f}'.format
# pd.set_option('display.max_colwidth', 5000)
#
#
# # In[25]:
#
#
# npt = nlplot.NLPlot(textblob_lemmatized, target_col="full_text")
# sw2 = npt.get_stopword(top_n=30, min_freq=0)
#
#
# # In[26]:
#
#
# npt.build_graph(stopwords=sw2, min_edge_frequency=25)
#
#
# # In[27]:
#
#
# display(npt.node_df.head(), npt.edge_df.head(),)
#
#
# # In[31]:
#
#
# npt.co_network(title="All Tesla Sentiment Co-Occurence Network",
#               color_palette="hls",
#               width=1000,
#               height=1200,)
#
#
# # In[29]:
#
#
# npt.sunburst(
#     title='All Tesla Sentiment Sunburst Chart',
#     colorscale=True,
#     color_continuous_scale='IceFire',
#     width=1000,
#     height=800,
# )
#
#
# # In[30]:
#
#
# npt.ldavis(num_topics=3, passes=5, save=False)
#

# In[ ]:
