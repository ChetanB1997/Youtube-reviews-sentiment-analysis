#Libraries 
import numpy as np
import pandas as pd 
# import matplotlib.pyplot as plt
import os

# Import functions for data preprocessing & data preparation
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import string
from string import punctuation
import re



def datatranformation(file_path):  
    data = pd.read_csv(file_path)
    return data


def labeling(data):
        
   # nltk.download('vader_lexicon')  #requirements
    le = LabelEncoder()
    sentiments = SentimentIntensityAnalyzer()
    data["Comment"] = data["Comment"].apply(str)
    data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Comment"]]
    data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Comment"]]
    data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Comment"]]
    data['Compound'] = [sentiments.polarity_scores(i)["compound"] for i in data["Comment"]]
    score = data["Compound"].values
    sentiment = []
    for i in score:
        if i >= 0.05 :
            sentiment.append('Positive')
        elif i <= -0.05 :
            sentiment.append('Negative')
        else:
            sentiment.append('Neutral')
    data["Sentiment"] = sentiment
    data.drop(['Positive','Negative','Neutral','Compound'],axis=1)
    data['Sentiment'] = le.fit_transform(data['Sentiment'])
    return data



def text_processing(text):  
    stop_words = stopwords.words('english')
    porter_stemmer = PorterStemmer()
    lancaster_stemmer = LancasterStemmer() 
    snowball_stemer = SnowballStemmer(language="english")
    lzr = WordNetLemmatizer()
 
    # convert text into lowercase
    text = text.lower()

    # remove new line characters in text
    text = re.sub(r'\n',' ', text)
    
    # remove punctuations from text
    text = re.sub('[%s]' % re.escape(punctuation), "", text)
    
    # remove references and hashtags from text
    text = re.sub("^a-zA-Z0-9$,.", "", text)
    
    # remove multiple spaces from text
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    
    # remove special characters from text
    text = re.sub(r'\W', ' ', text)

    text = ' '.join([word for word in word_tokenize(text) if word not in stop_words])
    
    # stemming using porter stemmer from nltk package - msh a7sn 7aga - momken: lancaster, snowball
    # text=' '.join([porter_stemmer.stem(word) for word in word_tokenize(text)])
    # text=' '.join([lancaster_stemmer.stem(word) for word in word_tokenize(text)])
    # text=' '.join([snowball_stemer.stem(word) for word in word_tokenize(text)])
    
    # lemmatizer using WordNetLemmatizer from nltk package
    text=' '.join([lzr.lemmatize(word) for word in word_tokenize(text)])

    return text


def processing(data):
    # nltk.download('omw-1.4')
    data.Comment = data.Comment.apply(lambda text: text_processing(text))
    processed_data = {
        'Sentence':data.Comment,
        'Sentiment':data['Sentiment']
    }
    processed_data = pd.DataFrame(processed_data)
    return processed_data



def sampling(processed_data):
    corpus = []
    processed_data['Sentiment'].value_counts()
    df_neutral = processed_data[(processed_data['Sentiment']==1)] 
    df_negative = processed_data[(processed_data['Sentiment']==0)]
    df_positive = processed_data[(processed_data['Sentiment']==2)]

    # upsample minority classes
    df_negative_upsampled = resample(df_negative, 
                                    replace=True,    
                                    n_samples= 205, 
                                    random_state=42)  

    df_neutral_upsampled = resample(df_neutral, 
                                    replace=True,    
                                    n_samples= 205, 
                                    random_state=42)  


    # Concatenate the upsampled dataframes with the neutral dataframe
    final_data = pd.concat([df_negative_upsampled,df_neutral_upsampled,df_positive])
    

    final_data['Sentiment'].value_counts()
    for sentence in final_data['Sentence']:
        corpus.append(sentence)
        
    cv = CountVectorizer(max_features=1500)
    Xdata = cv.fit_transform(corpus).toarray()
    Ydata = final_data.iloc[:, -1].values
    # print(final_data)
    
    path = 'data/final_result.csv'
    # with open(path, "w") as file:
    #     file.write(final_data)
    final_data.to_csv(path,  index=False)
    # # path = 'data/f_data.csv'
       
    return Xdata,Ydata

if __name__ == "__main__":
    
    labeling()
    text_processing()
    processing()
    sampling()