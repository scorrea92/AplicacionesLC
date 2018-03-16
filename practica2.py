
import csv
import pandas as pd
import xml.etree.ElementTree as ET
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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

#"""Tweet Tokenize"""
#tknzr = TweetTokenizer()
#path = 'data/TASS2017_T1_training.xml'
#
#from bs4 import BeautifulSoup
#infile = open(path,"r")
#contents = infile.read()
#soup = BeautifulSoup(contents,'xml')
#content = soup.find_all('content')
#
#train=[]
#for c in content:
#    train.append(' '.join(tknzr.tokenize(c.getText())))
#

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


"""SVM"""
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









