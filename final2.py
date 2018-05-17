#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 14:49:05 2018

@author: sebastiancorrea
"""

import numpy as np
import csv
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import f1_score

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


path_X = '../datasets/AggressiveDetection_training/AggressiveDetection_train.txt'
path_Y = '../datasets/AggressiveDetection_training/AggressiveDetection_train_solution.txt'

aux_x = tcv2array(path_X)
y = np.loadtxt(path_Y)

aux_x[0]

"""Solo correr esto para obtener resultados reales"""
# Tweet Tokenize
tknzr = TweetTokenizer()

x = []
for i in aux_x:
     x.append(' '.join(tknzr.tokenize(i[0])))

### Transform data for model
le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

cv = TfidfVectorizer()
x = cv.fit_transform(x)
x = x.toarray()

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1992)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)



"""SVM"""
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, recall_score, precision_score

svm = LinearSVC()
svm.fit(x_train, y_train)  


y_pred = svm.predict(x_test)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(y_test, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(y_test, y_pred))


print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='weighted'))
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))




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


print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='weighted'))
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))




#Naive Bayes Gaussian
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb = gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(y_test, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(y_test, y_pred))


print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='weighted'))
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))





#KNeighbors
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=100)

neigh = neigh.fit(x_train, y_train)

y_pred = neigh.predict(x_test)
print("______________Validation Confusion Matrix______________")
print(confusion_matrix(y_test, y_pred))
print("")
print("___________________Validation Report___________________")
print(classification_report(y_test, y_pred))


print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='weighted'))
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))





#Neural Network
from keras import backend as K
from keras.models import Sequential 
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization as BN
from keras.layers import GaussianNoise as GN
from keras.optimizers import Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], x_train.shape[1])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

num_classes = 1
batch_size = 32
epochs = 100
learnRate = 0.001

# Learning rate annealing
def step_decay(epoch):
    if epoch/epochs<0.3:
        lrate = learnRate
    elif epoch/epochs<=0.5:
        lrate = learnRate/2
    elif epoch/epochs<=0.70:
        lrate = learnRate/10
    else:
        lrate = learnRate/100
    return lrate

#Loss function for macro_fm
def macro_fm(y_true, y_pred, beta=1.0):
    beta2 = beta**2.0
    top = K.sum(y_true * y_pred, axis=0)
    bot = beta2 * K.sum(y_true, axis=0) + K.sum(y_pred, axis=0)
    return -(1.0 + beta2) * K.mean(top/bot)
  
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
model.summary()

checkpoint_path = "Wehigts.hdf5"
checkpointer = ModelCheckpoint(filepath=checkpoint_path,
                               monitor='val_acc', verbose=1,
                               save_best_only=True, mode='max')


loss = macro_fm#'binary_crossentropy'

adam = Adam(lr=learnRate, beta_1=0.9, beta_2=0.999,
            epsilon=None, decay=1e-6, amsgrad=False)

rms = RMSprop(lr=learnRate, rho=0.9, epsilon=None, decay=0.0)

lrate = LearningRateScheduler(step_decay)

model.compile(loss=loss, 
            optimizer=adam, 
            metrics=['accuracy']) 

history = model.fit(x_train, y_train, 
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[checkpointer])

#Load best model
model.load_weights(checkpoint_path)

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

print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='weighted'))
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))

from keras.utils import plot_model
plot_model(model, to_file='model.png',show_shapes=True)

