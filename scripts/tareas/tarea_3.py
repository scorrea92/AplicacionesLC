import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw

def simplified_lesk(word,sentence):
    sentidos=wn.synsets(word)
    best_sense = sentidos[0]
    max_overlap = 0
    context = nltk.word_tokenize(sentence)

    for sense in sentidos:
        signatura = [w.lower() for w in nltk.word_tokenize(sense.definition()) if w.lower() not in sw_english]
        for ex in sense.examples():
            signatura += [w.lower() for w in nltk.word_tokenize(ex) if w.lower() not in sw_english]
        overlap = len(list(set(signatura).intersection(context)))
        # print(overlap)
        # print(sense.definition())
        # print(sense.examples())
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    return best_sense

sw_english=sw.words('english')
sentence = "Yesterday I went to the bank to withdraw the money and the credit card did not work"
word = "bank"
print(simplified_lesk(word,sentence))

