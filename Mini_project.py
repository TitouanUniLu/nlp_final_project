import nltk                                # Python library for NLP
import random     
import pandas as pd                         # pseudo-random number generator
import re                                  # library for regular expression operations
import string                              # for string operations
import numpy as np
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import TweetTokenizer   # module for tokenizing strings
from os.path import exists
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


def clean(unclean): 
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, # Initializing the tokenizer from nltk
                                reduce_len=True)
    stopwords_english = stopwords.words('english') #importing english stopwords 
    stemmer = PorterStemmer() #Initializing the stemmer class 
    clean_text = []
    clean_stem_text = [] 

    # remove stock market tickers like $GE
    unclean = re.sub(r'\$\w*', '', unclean)
    # Remove old style retweet text "RT"
    unclean = re.sub(r'^RT[\s]+', '', unclean)
    # Remove \n characters 
    unclean = unclean.strip()
    # Remove hyperlinks
    unclean = re.sub(r'https?:\/\/.*[\r\n]*', '', unclean)
    # Remove hashtags
    # only removing the hash # sign from the word
    unclean = re.sub(r'#', '', unclean)
    # Tokenizing
    unclean = tokenizer.tokenize(unclean)
    for word in unclean: # Go through every word in your tokens list
        if (word not in stopwords_english and  word not in string.punctuation):  # remove punctuation and stopwords
            clean_text.append(word)

    for word in clean_text:
        stem_word = stemmer.stem(word)  # stemming word
        clean_stem_text.append(stem_word)  # append to the list

    return clean_stem_text 

def clean_text():
    if exists('clean_text_news.csv'): 
        return  pd.read_csv(r'clean_text_news.csv')
    else: 
        df = pd.read_csv(r'news.csv')
        df['text']=df['text'].apply(clean)
        df['title']=df['title'].apply(clean)
        df.to_csv('clean_text_news.csv', index=False)
        return df

def multinomial (X_train_tf,X_test_tf, y_test, y_train): 
    mnb = MultinomialNB()
    mnb.fit(X_train_tf, y_train)
    y_pred = mnb.predict(X_test_tf)
    mnb_score = accuracy_score(y_test, y_pred) 
    print("Naïve Bayes Classifier")
    print("Accuracy score is: ",mnb_score)
    print(classification_report(y_test, y_pred))
    print("----------------------------------------------------------------------")
    return 0 

def logistic_regression(X_train_tf,X_test_tf, y_test, y_train):
    lr = LogisticRegression()
    lr.fit(X_train_tf, y_train)
    y_pred = lr.predict(X_test_tf)
    score = lr.score(X_test_tf, y_test)
    print("Logistic regression")
    print("Accuracy score is: ",score)
    print(classification_report(y_test, y_pred))
    print("----------------------------------------------------------------------")
    return 0 

def support_vector_machines(X_train_tf,X_test_tf, y_test, y_train):
    clf = svm.SVC(kernel='linear') # Linear Kernel
    clf.fit(X_train_tf, y_train)
    y_pred = clf.predict(X_test_tf)
    score = clf.score(X_test_tf, y_test)
    print("Support Vector Machines")
    print("Accuracy score is: ",score)
    print(classification_report(y_test, y_pred))
    print("----------------------------------------------------------------------")
    return 0 

def k_neighbour(X_train_tf,X_test_tf, y_test, y_train): 
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_tf, y_train)
    y_pred = knn.predict(X_test_tf)
    score = knn.score(X_test_tf, y_test)
    print("K-nearest-neighbors with K = 5")
    print("Accuracy score is: ",score)
    print(classification_report(y_test, y_pred))
    print("----------------------------------------------------------------------")
    return 0 

def main(): 
    df = clean_text()
    x = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    tf_vectorizer = TfidfVectorizer(use_idf=True) 
    X_train_tf = tf_vectorizer.fit_transform(X_train)
    X_test_tf = tf_vectorizer.transform(X_test)
    multinomial(X_train_tf,X_test_tf, y_test, y_train)
    logistic_regression(X_train_tf,X_test_tf, y_test, y_train)
    support_vector_machines(X_train_tf,X_test_tf, y_test, y_train)
    k_neighbour(X_train_tf,X_test_tf, y_test, y_train)
    



if __name__ == "__main__":
    main()