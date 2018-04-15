# -*- coding: utf-8 -*-
import csv
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing

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
    
    return contents,  df['polarity']

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
train_x, train_y = xml2df(path)

path1 = 'data/TASS2017_T1_development.xml'
dev_x, dev_y = xml2df(path1)

# path2 = 'data/TASS2017_T1_test.xml'
# test_df = xml2df(path2)

#"""Read Dictionary"""
#path = 'data/ElhPolar_esV1.lex'
#dicc_full = tcv2array(path)
#dicc = list(pd.DataFrame(dicc_full)[0])

"""Transform data for model"""
le = preprocessing.LabelEncoder()
train_y = le.fit_transform(train_y)
dev_y = le.transform(dev_y)

cv = TfidfVectorizer()
train_x = cv.fit_transform(train_x)
dev_x = cv.transform(dev_x)


"""Linear SVM"""
print("""Linear SVM""")
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, classification_report

svm = LinearSVC()
svm.fit(train_x, train_y)  


y_pred = svm.predict(dev_x)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(dev_y, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(dev_y, y_pred))



from sklearn.model_selection import GridSearchCV

param_grid = {'C': [1, 10, 100, 1000]}

clf = GridSearchCV(LinearSVC(), param_grid, scoring='f1_micro')
clf = clf.fit(train_x, train_y)

print("Best estimator found by grid search: {}".format(clf.best_estimator_))
print("Tuned Logistic Regression Accuracy: {}".format(clf.best_score_))

y_pred = clf.predict(dev_x)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(dev_y, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(dev_y, y_pred))




"""SVM"""
print("""SVM""")
from sklearn import svm
# Set the parameters by cross-validation
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                'C': [1, 10, 100, 1000]}]

print("# Tuning hyper-parameters")

clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, scoring='f1_micro')
clf.fit(train_x, train_y)

print("Best estimator found by grid search: {}".format(clf.best_estimator_))
print("Tuned Logistic Regression Accuracy: {}".format(clf.best_score_))

y_pred = clf.predict(dev_x)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(dev_y, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(dev_y, y_pred))




"""DecisionTreeClassifier"""
print("""DecisionTreeClassifier""")
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_x, train_y)

y_pred = clf.predict(dev_x)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(dev_y, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(dev_y, y_pred))



"""Naive Bayes Gaussian"""
print("""Naive Bayes Gaussian""")
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb = gnb.fit(train_x.toarray(), train_y)

y_pred = gnb.predict(dev_x.toarray())
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(dev_y, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(dev_y, y_pred))



"""KNeighbors"""
print("""KNeighbors""")
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)

neigh = neigh.fit(train_x.toarray(), train_y)

y_pred = neigh.predict(dev_x.toarray())
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(dev_y, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(dev_y, y_pred))

parameters = {'n_neighbors': [10, 500, 100, 150, 200, 250]}

print("# Tuning hyper-parameters")

clf = GridSearchCV(KNeighborsClassifier(), parameters, scoring='f1_micro')
clf.fit(train_x, train_y)

print("Best estimator found by grid search: {}".format(clf.best_estimator_))
print("Tuned Logistic Regression Accuracy: {}".format(clf.best_score_))

y_pred = clf.predict(dev_x.toarray())
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(dev_y, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(dev_y, y_pred))




