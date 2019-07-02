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

def data_transform(aux_x):
    
    #Antes de tokenizado
    
    #Conteo de mayusculas
    
    conteo_mayusculas = []
    for i in aux_x:
      
      aux = i[0].replace("@USUARIO","").replace(" ","")
      total_character = len([w for w in aux if w.isalpha()])
      
      if total_character: 
        conteo_mayusculas.append( len([w for w in aux if w.isupper()])/total_character )
      else:
        conteo_mayusculas.append(0)
    conteo_mayusculas = np.array(conteo_mayusculas)
    #simbolos ¡!¿?
    #insultos
    """
    ::insultos::
    
    Hijo de puta, hijo puta, hjo puta, hijo de la gran puta, puta madre, chinga tu madre, hijo de perra, hijo de la chingada, hijo de tu pinche madre, hijo de la rechingada, hijo de tu puta madre, hijo de tu reputisima madre
    -> hdp
    
    pura mierda -> pura_mierda
    (opcional: chingo, puta) su/tu madre ->  su_madre
    
    mi madre -> mi_madre
    
    después de tokenizar
    
    ::Insuto en Hashtag::
    
    si en #_ aparece JOTO, PERRO, PUTO, PUTA, CHINGA, VERGA, PUTIZA, CULERO, CABRON, PENDEJO, FELON, EMPUTADO, MARICON, GRINGO
    """
    
    for aux in aux_x:
      tempStr = aux[0].lower().replace("‼","! ! ").replace("!","! ").replace("¡","¡ ").replace("?","? ").replace("¿","¿ ")
      #insultos - hijo de puta
      tempStr = tempStr.replace("hijo de puta","hdp").replace("hijo puta","hdp").replace("hijo de la gran puta","hdp")
      tempStr = tempStr.replace("chinga tu madre","hdp").replace("hijo de perra","hdp").replace("hijo de la chingada","hdp")
      tempStr = tempStr.replace("hijo de tu pinche madre","hdp").replace("hijo de la rechingada","hdp").replace("hijo de tu puta madre","hdp")
      tempStr = tempStr.replace("hijo de tu reputa madre","hdp").replace("hijo de tu reputisima madre","hdp")
      
      tempStr = tempStr.replace("hija de puta","hdp").replace("hija puta","hdp").replace("hija de la gran puta","hdp")
      tempStr = tempStr.replace("hija de perra","hdp").replace("hija de la chingada","hdp")
      tempStr = tempStr.replace("hija de tu pinche madre","hdp").replace("hija de la rechingada","hdp").replace("hija de tu puta madre","hdp")
      tempStr = tempStr.replace("hija de tu reputa madre","hdp").replace("hija de tu reputisima madre","hdp")
      
      tempStr = tempStr.replace("tu madre","tu_madre").replace("su madre","su_madre")
      tempStr = tempStr.replace("pura mierda","pura_mierda")
      
      aux[0] = tempStr
    
    
    #Después de tokenizado
    tknzr = TweetTokenizer()
    
    x_tok = []
    for i in aux_x:
         x_tok.append(tknzr.tokenize(i[0]))
    
    #Control de risas
    
    for s in x_tok:
    
      for i in range(len(s)): # s = token_x
        
        #risas
        
        j_s = s[i].count('j')
        h_s = s[i].count('h')
        a_s = s[i].count('a')+s[i].count('s')
        e_s = s[i].count('e')
        i_s = s[i].count('i')
        o_s = s[i].count('o')
        u_s = s[i].count('u')
        #no considera risa algo como aaaahh, ha, haaa, ah etc, minimo un intercalado de grado 2
        laugh = False
        if h_s and not j_s:
          laugh = True
          auxCount = 0
          for j in range(1,len(s[i])):
            if not s[i][j] == s[i][j-1]:
              auxCount += 1
          if auxCount == 1:
            laugh = False
        
        if j_s or laugh: #solo si hay js o hs, si no sería aaaa, eeee, iii y eso no es risa
          j_s+=h_s
          if len(s[i]) == j_s+a_s:
            s.pop(i)
            for j in range(max(j_s,a_s)):
              s.insert(i,'ja')
              i+=1
          elif len(s[i]) == j_s+e_s:
            s.pop(i)
            for j in range(max(j_s,e_s)):
              s.insert(i,'je')
              i+=1
          elif len(s[i]) == j_s+i_s:
            s.pop(i)
            for j in range(max(j_s,i_s)):
              s.insert(i,'ji')
              i+=1
          elif len(s[i]) == j_s+o_s:
            s.pop(i)
            for j in range(max(j_s,o_s)):
              s.insert(i,'jo')
              i+=1
          elif len(s[i]) == j_s+u_s:
            s.pop(i)
            for j in range(max(j_s,u_s)):
              s.insert(i,'ju')
              i+=1
              
        #Hash, como pueden afectar a un conjunto de tweets similares, por ahora mejor no tocarlo
        
        
    #print(x_tok[2282])
    x = []
    for i in x_tok:
         x.append(' '.join(i))
    
    return x, conteo_mayusculas

def data_to_model(path_x, path_y, path_test):
    aux_x = tcv2array(path_x)
    aux_x_test = tcv2array(path_test)
    y = np.loadtxt(path_y)
    
    x, conteo_mayusculas = data_transform(aux_x)
    
    """Transform data for model"""
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)
    
    cv = TfidfVectorizer()
    x = cv.fit_transform(x)
    x = x.toarray()
    x = np.insert(x, -1, conteo_mayusculas, axis =1)
    
    """Test Data"""
    x_test, conteo_mayusculas_test = data_transform(aux_x_test)
    x_test = cv.transform(x_test)
    x_test = x_test.toarray()
    x_test = np.insert(x_test, -1, conteo_mayusculas_test, axis =1)
    
    
    print(x.shape)
    print(x_test.shape)
    
    return x, y, x_test
    
from sklearn.model_selection import train_test_split

path_X = '../datasets/AggressiveDetection_training/AggressiveDetection_train.txt'
path_Y = '../datasets/AggressiveDetection_training/AggressiveDetection_train_solution.txt'
path_test = '../datasets/AggressiveDetection_training/AggressiveDetection_test.txt'

x , y, x_test_d = data_to_model(path_X, path_Y, path_test)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1992)

from sklearn import decomposition
pca = decomposition.PCA(n_components=0.99, svd_solver='full')
pca.fit(x_train)
x_train = pca.transform(x_train)
x_test = pca.transform(x_test)
x_test_d = pca.transform(x_test_d)

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

from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='weighted'))
print(accuracy_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(precision_score(y_test, y_pred))


"""Predict Values"""
y_final = svm.predict(x_test_d)

out =""
for i in range(len(y_final)):
    out += "aggressiveness"+"\t"+ "tweet-"+str(i+1)+ "\t" +str(y_final[i])+"\n"

f= open("SEAL-UPV_Aggressive_Detection.txt","w+")
f.write(out)
f.close()


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

from sklearn.metrics import f1_score
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

from sklearn.metrics import f1_score
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

from sklearn.metrics import f1_score
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

