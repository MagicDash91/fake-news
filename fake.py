import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

st.header('Fake News Detection')

df = pd.read_csv('fake_or_real_news.csv.zip')

df['label'] = df['label'].replace(['FAKE'],'0')
df['label'] = df['label'].replace(['REAL'],'1')

df['label'] = pd.to_numeric(df['label'])

X = df['title']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
tfid_x_train = tfvect.fit_transform(X_train)
tfid_x_test = tfvect.transform(X_test)

classifier = PassiveAggressiveClassifier(max_iter=200)
classifier.fit(tfid_x_train,y_train)


input_data = st.text_input('Input your News title here')
input_data=[input_data]
vectorized_input_data = tfvect.transform(input_data)
prediction = classifier.predict(vectorized_input_data)


if prediction == '[0]':
  prediction2 = prediction.replace('[0]', 'FAKE NEWS')
  prediction3 = prediction2.astype(string)
elif prediction == '[1]':
  prediction2 = prediction.replace('[1]', 'REAL NEWS')
  prediction3 = prediction2.astype(string)

st.info(prediction3)
