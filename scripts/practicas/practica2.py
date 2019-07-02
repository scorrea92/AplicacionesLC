# -*- coding: utf-8 -*-
import csv
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

from sklearn.svm import LinearSVC
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report
)

def xml2df(path):
    """Function for transform xml dataset to pandas data frame con tweet tok"""
    XMLdata = open(path).read()

    XML = ET.XML(XMLdata)

    all_records = []
    for i, child in enumerate(XML):
        record = {}
        for subchild in child:
            record[subchild.tag] = subchild.text
            for subsubchild1 in subchild:
                for subsubchild in subsubchild1:
                    record[subsubchild.tag] = subsubchild.text
                
        all_records.append(record)
    df = pd.DataFrame(all_records)
    df = df.rename(columns={ 'value' : 'polarity'})
    del df['sentiment']

    """Tweet Tokenize"""
    tknzr = TweetTokenizer()

    contents = []
    for content in df['content']:
        contents.append(' '.join(tknzr.tokenize(content)))
    
    return contents,  df['polarity'], df

def tcv2array(path):
    """Read tab separated values, # is for comments and dont be load it"""
    a = []
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if row:
                if row[0][0] != '#':
                    a.append(row)
    return a

"""Read Data to Data Frame"""
path = 'data/TASS2017_T1_training.xml'
train_x, train_y, _ = xml2df(path)

path1 = 'data/TASS2017_T1_development.xml'
dev_x, dev_y, _ = xml2df(path1)


"""Transform data for model"""
le = preprocessing.LabelEncoder()
train_y = le.fit_transform(train_y)
dev_y = le.transform(dev_y)
y = np.append(dev_y, train_y)


cv = TfidfVectorizer()
x = np.append(dev_x, train_x)
train_x = cv.fit_transform(train_x)
dev_x = cv.transform(dev_x)
x = cv.transform(x)

"""Linear SVM"""
print("""Linear SVM""")
svm = LinearSVC()
svm.fit(train_x, train_y)  


y_pred = svm.predict(dev_x)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(dev_y, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(dev_y, y_pred))


print(f1_score(dev_y, svm.predict(dev_x), average='micro'))
print(f1_score(dev_y, svm.predict(dev_x), average='macro'))
print(f1_score(dev_y, svm.predict(dev_x), average='weighted'))

"""Final Model"""
svm = LinearSVC()
svm.fit(x, y)

path2 = 'data/TASS2017_T1_test.xml'
test, _, ids = xml2df(path2)
test = cv.transform(test)

y_pred = svm.predict(test)
y_pred_text = ""
for i in y_pred:
    y_pred_text += str(i) + "\n"
    
f= open("data/Sebastian_Correa_Linear_SVM_Num.txt","w+")
f.write(y_pred_text)
f.close()

y_pred  = le.inverse_transform(y_pred)
y_pred_text = ""
for index, i in enumerate(y_pred):
    y_pred_text += ids['tweetid'][index]+"    "+i + "\n"
    
f= open("data/Sebastian_Correa_Linear_SVM_Cat.txt","w+")
f.write(y_pred_text)
f.close()
