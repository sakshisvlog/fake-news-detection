#Importing the libraries
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import numpy as np
import matplotlib.pyplot as plt
import numpy as np1
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plot
import cv2 as cv

#Importing the cleaned file containing the text and label
news = pd.read_csv('news.csv')
X = news['text']
y = news['label']

#Splitting the data into train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Creating a pipeline that first creates bag of words(after applying stopwords) & then applies Multinomial Naive Bayes model
pipeline1 = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('nbmodel', MultinomialNB())])
                    
#Creating a pipeline that first creates bag of words(after applying stopwords) & then applies Multinomial Naive Bayes model
pipeline2 = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                    ('nbmodel', KNeighborsClassifier())])

#Training our data
pipeline1.fit(X_train, y_train)
pipeline2.fit(X_train, y_train)

#Predicting the label for the test data
pred1 = pipeline1.predict(X_test)
pred2 = pipeline2.predict(X_test)



#Checking the performance of our model
print(classification_report(y_test, pred1))
print(metrics.confusion_matrix(y_test, pred1))
print(metrics.accuracy_score(y_test, pred1))

#Checking the performance of our model
print(classification_report(y_test, pred2))
print(metrics.confusion_matrix(y_test, pred2))
print(metrics.accuracy_score(y_test, pred2))


#Serialising the file
with open('model.pickle1', 'wb') as handle:
    pickle.dump(pipeline1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('model.pickle2', 'wb') as handle:
    pickle.dump(pipeline2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
N= 3
ind = np.arange(N) 
plt.xticks(ind, ('Precision', 'Recall', 'F1-Score')) 

data1 = [78,98, 86]
data2 = [99, 13, 23]
width =0.3
plt.xlabel("KNN vs NB") 
plt.ylabel("Prediction Scores") 
plt.bar(np.arange(len(data1)), data1, width=width)
plt.bar(np.arange(len(data2))+ width, data2, width=width)
plt.title('Classification Summary of KNN-NB for Real news:')
plt.show()



ind1 = np1.arange(N) 
plt1.xticks(ind1, ('Precision', 'Recall', 'F1-Score')) 

data3 = [97,71, 82]
data4 = [56, 99, 71]
plt.xlabel("KNN vs NB") 
plt.ylabel("Prediction Scores") 
plt.bar(np1.arange(len(data3)), data3, width=width)
plt.bar(np1.arange(len(data4))+ width, data4, width=width)
plt.title('Classification Summary of KNN-NB for Fake news:')
plt.show()

