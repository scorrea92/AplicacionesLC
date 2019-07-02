# coding: utf-8
import nltk
from nltk.corpus import cess_esp
from random import shuffle

corpus = cess_esp.tagged_sents()

number_sentences=len(corpus)

prueba = corpus
corpusnew = []
for sen in prueba:
	frase = []
	for i in range(0,len(sen)):
		if not(sen[i][0] == '*0*'):
			if sen[i][1][0]=='F' or sen[i][1][0]=='v':
				final = 3
			else:
				final = 2

			frase.append( tuple( [ sen[i][0],sen[i][1][:final] ] ) )
	corpusnew.append(frase)


corpus = corpusnew
# shuffle(corpus)

#Creo train y tes al 90/10
train=corpus[:9*number_sentences//10]
test=corpus[9*number_sentences//10:]

#Problema entrenamiento 90% test 10%
print("Problema entrenamiento 90% test 10%")

#modelo Markov
from nltk.tag import  hmm
trainer = hmm.HiddenMarkovModelTrainer()
tagger_markov = trainer.train_supervised(train)

print("Modelo de Markov ", tagger_markov.evaluate(test))


# #modelo TnT
# from nltk.tag import  tnt
# tagger_tnt = tnt.TnT()
# tagger_tnt.train(train)


# print("Modelo TnT ",tagger_tnt.evaluate(test))



# print("Modelo TnT suavizado ",tagger_tnt.evaluate(test))

# #Problema entrenamiento cross validation particiones 10
# print("Problema cross validation particiones 10")


# #modelo Markov
# print("Modelo Markov")
# sec = number_sentences//10

# error_total = 0
# for i in range(0,10):
# 	test = corpus[i*sec:(i+1)*sec]
# 	train = [i for i in corpus if i not in test]
# 	trainer = hmm.HiddenMarkovModelTrainer()
# 	tagger_markov = trainer.train_supervised(train)
# 	print(tagger_markov.evaluate(test))
# 	error_total += tagger_markov.evaluate(test)
	
# print("Acierto total: ", error_total/10)


# #modelo TnT
# print("Modelo TnT")
# from nltk.tag import  tnt
# error_total = 0
# for i in range(0,10):
# 	test = corpus[i*sec:(i+1)*sec]
# 	train = [i for i in corpus if i not in test]
# 	tagger_tnt = tnt.TnT()
# 	tagger_tnt.train(train)
# 	print(tagger_tnt.evaluate(test))
# 	error_total += tagger_tnt.evaluate(test)

# print("Acierto total final: ", error_total/10)









