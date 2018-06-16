# -*- coding: utf-8 -*-
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.externals import joblib

#import the dataset 
df = pd.read_csv('imdb_labelled.txt', delimiter = '\t', engine='python', quoting = 3)

#cleaning the dataset
import re 
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Create BOW model
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfVectorizer = TfidfVectorizer(max_features =2000)
X = tfidfVectorizer.fit_transform(corpus).toarray()
y = df.iloc[:, 1].values


from sklearn.model_selection import train_test_split 
X_train, X_test , y_train, y_test = train_test_split(X, y , test_size = 0.20)


#Fit Naive Bayes to the training set 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train) 
 

 
#Predicting the test set results 
predictions = classifier.predict(X_test)

#Making the confusion matrix 
from sklearn.metrics import confusion_matrix,classification_report
cm = confusion_matrix(y_test, predictions) 
cr = classification_report(y_test,predictions)

joblib.dump(classifier, 'classifier.pkl')
joblib.dump(tfidfVectorizer, 'tfidfVectorizer.pkl')

