# -*- coding: utf-8 -*-
import numpy as np
from keras import backend as K
import csv
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report


def tcv2array(path):
    """Read tab separated values, # is for comments and dont be load it"""
    a = []
    with open(path) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        for row in reader:
            if row:
                if row[0][0]:
                    a.append(row)
    return a


aux_x = tcv2array('../datasets/AggressiveDetection_training/AggressiveDetection_train.txt')
y = np.loadtxt('../datasets/AggressiveDetection_training/AggressiveDetection_train_solution.txt')


"""Tweet Tokenize"""
tknzr = TweetTokenizer()

x = []
for i in aux_x:
     x.append(' '.join(tknzr.tokenize(i[0])))



"""Transform data for model"""
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

cv = TfidfVectorizer()
x = cv.fit_transform(x)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1992)


"""SVM"""
from sklearn.svm import LinearSVC

svm = LinearSVC()
svm.fit(x_train, y_train)  


y_pred = svm.predict(x_test)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(y_test, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(y_test, y_pred))



"""DecisionTreeClassifier"""
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(y_test, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(y_test, y_pred))



"""Naive Bayes Gaussian"""
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb = gnb.fit(x_train.toarray(), y_train)

y_pred = gnb.predict(x_test.toarray())
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(y_test, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(y_test, y_pred))



"""KNeighbors"""
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=10)

neigh = neigh.fit(x_train.toarray(), y_train)

y_pred = neigh.predict(x_test.toarray())
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(y_test, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(y_test, y_pred))



def macro_fm(y_true, y_pred, beta=1.0):
    beta2 = beta**2.0
    top = K.sum(y_true * y_pred, axis=0)
    bot = beta2 * K.sum(y_true, axis=0) + K.sum(y_pred, axis=0)
    return -(1.0 + beta2) * K.mean(top/bot)

def micro_fm(y_true, y_pred):
    """Used with softmax this is equivalent to accuracy
    Two other commonly used F measures are the F2, which weighs recall
    higher than precision (by placing more emphasis on false negatives),
    and the F0.5, which weighs recall lower than precision (by
    attenuating the influence of false negatives).
    With beta = 1, this is equivalent to a F-measure. With beta < 1,
    assigning correct classes becomes more important, and with beta >
    1 the metric is instead weighted towards penalizing incorrect
    class assignments.
    """
    beta = 0.1
    beta2 = beta**2.0
    top = K.sum(y_true * y_pred)
    bot = beta2 * K.sum(y_true) + K.sum(y_pred)
    return -(1.0 + beta2) * top / bot

"""Neural Network"""
from keras.models import Sequential 
from keras.layers import Dense, Activation	
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN

x_train = x_train.toarray().reshape(x_train.shape[0], x_train.shape[1])
x_test = x_test.toarray().reshape(x_test.shape[0], x_train.shape[1])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

num_classes = 1
batch_size = 32
epochs = 100


#Define model architecture
model = Sequential()
model.add( Dense( 2048, activation='relu', input_shape=(x_train.shape[1],) ) )
model.add(BN())
model.add(GN(0.3))
model.add(Activation('relu'))

model.add(Dense(1024))
model.add(BN())
model.add(GN(0.3))
model.add(Activation('relu'))

model.add(Dense(512))
model.add(BN())
model.add(GN(0.3))
model.add(Activation('relu'))

model.add(Dense(256))
model.add(BN())
model.add(GN(0.3))
model.add(Activation('relu'))

model.add(Dense(num_classes, activation='sigmoid'))

#softplus softsign softmax elu selu relu tanh sigmoid hard_sigmoid linear
model.summary()

loss = macro_fm
#loss = 'binary_crossentropy'

model.compile(loss=loss, 
            optimizer='adam', 
            metrics=['accuracy']) 

history = model.fit(x_train, y_train, 
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


y_pred = model.predict(x_test, batch_size=1)
y_pred = np.where(y_pred > 0.5, 1, 0)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(y_test, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(y_test, y_pred))

