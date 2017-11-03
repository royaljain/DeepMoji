import json
from deepmoji.model_def import deepmoji_transfer, load_specific_weights
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH, NB_TOKENS
from deepmoji.class_avg_finetuning import class_avg_finetune
from deepmoji.finetuning import calculate_batchsize_maxlen
from deepmoji.sentence_tokenizer import SentenceTokenizer
from sklearn.model_selection import train_test_split

import os

absPath = os.path.dirname(os.path.dirname(__file__))
relPath = 'data/Emobank/'

dataPath = os.path.join(absPath, relPath)


print 'DATAPATH : ', dataPath

class DeepmojiEncoding:

    def __init__(self, X, y, test_X, test_Y, numEpoch = 100):

        print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))

        vocabulary = {}
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)

        batchsize,  maxlen = calculate_batchsize_maxlen(X)

        print 'MaxLen: ', maxlen
        print 'BatchSize: ', batchsize

        self.__st = SentenceTokenizer(vocabulary, maxlen)

        print('Loading model from {}.'.format(PRETRAINED_PATH))
        self.__model = deepmoji_transfer(nb_classes=2, maxlen=maxlen, weight_path=PRETRAINED_PATH )


        texts_unicode = [s.decode('utf-8') for s in X]
        tokenized, _, _ = self.__st.tokenize_sentences(texts_unicode)

        texts_unicode_test = [s.decode('utf-8') for s in test_X]
        tokenized_test, _, _ = self.__st.tokenize_sentences(texts_unicode_test)



        X_train, X_val, y_train, y_val = train_test_split(tokenized, y, test_size=0.2, random_state=42)

        X_test = tokenized_test
        y_test = test_Y

        print len(y)
        print len(y_train)
        print len(y_val)
        print len(y_test)

        data_texts = [X_train, X_val, X_test]
        data_labels = [y_train, y_val, y_test]


        self.__model, f1 = class_avg_finetune(self.__model, data_texts, data_labels,
                                       nb_classes=5, nb_epochs=1, batch_size= batchsize, method='chain-thaw',dataPath=dataPath)


        print(self.__model.summary())


    def getEncoding(self, listOfSentences):
        texts_unicode = [s.decode('utf-8') for s in listOfSentences]
        tokenized, _, _ = self.__st.tokenize_sentences(texts_unicode)
        #print('Encoding texts..')
        encoding = self.__model.predict(tokenized)
        return encoding





import pandas as pd
import csv

df = pd.read_csv(dataPath+'EmobankTrain.tsv', sep='\t', quoting=csv.QUOTE_NONE, error_bad_lines=False)

df['valence_class'] = df['Valence'].map(lambda x: int(x*4.9999))

X = df['sentence'].as_matrix()
Y = df['valence_class'].as_matrix()

print df.groupby('valence_class').count()

from keras.utils import to_categorical

Y = to_categorical(Y)


df_test = pd.read_csv(dataPath+'EmobankTest.tsv', sep='\t', quoting=csv.QUOTE_NONE, error_bad_lines=False)

df_test['valence_class'] = df_test['Valence'].map(lambda x: int(x*4.9999))

print df_test.groupby('valence_class').count()

test_X = df_test['sentence'].as_matrix()
test_Y = df_test['valence_class'].as_matrix()

test_Y = to_categorical(test_Y)


print len(X)
print len(test_X)
print len(Y)
print len(test_Y)


print Y[0:5]
print test_Y[0:5]
print X[0]
print test_X[0]

DeepmojiEncoding(X,Y,test_X,test_Y)



