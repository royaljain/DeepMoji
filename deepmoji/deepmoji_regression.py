#import sys
#sys.path.insert(0, '/Users/royal/Desktop/deepmoji/DeepMoji/')

import json
from deepmoji import class_avg_finetuning
reload(class_avg_finetuning)
from deepmoji.model_def import deepmoji_architecture, load_specific_weights
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH, NB_TOKENS
#from deepmoji.class_avg_finetuning import class_avg_finetune
from deepmoji.finetuning import calculate_batchsize_maxlen
from deepmoji.sentence_tokenizer import SentenceTokenizer
from sklearn.model_selection import train_test_split

dataPath = '../data/'


class DeepmojiEncoding:

    def __init__(self, X, y, numEpoch = 100):

        print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))

        vocabulary = {}
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)

        batchsize,  maxlen = calculate_batchsize_maxlen(X)

        print 'MaxLen: ', maxlen
        print 'BatchSize: ', batchsize

        self.__st = SentenceTokenizer(vocabulary, maxlen)

        print('Loading model from {}.'.format(PRETRAINED_PATH))

        self.__model = deepmoji_architecture(nb_classes=1, nb_tokens=NB_TOKENS,
                                      maxlen=maxlen, feature_output=False,
                                      return_attention=False)

        load_specific_weights(self.__model, PRETRAINED_PATH, exclude_names=['softmax'])

        self.__st = SentenceTokenizer(vocabulary, maxlen)

        texts_unicode = [s.decode('utf-8') for s in X]
        tokenized, _, _ = self.__st.tokenize_sentences(texts_unicode)

        X_train, X_test, y_train, y_test = train_test_split(tokenized, y, test_size=0.1, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


        print len(y)
        print len(y_train)
        print len(y_val)
        print len(y_test)

        data_texts = [X_train, X_val, X_test]
        data_labels = [y_train, y_val, y_test]


        self.__model, f1 = class_avg_finetuning.class_avg_finetune(self.__model, data_texts, data_labels,
                                       1, nb_epochs=1, batch_size= batchsize, loss = 'mean_squared_error', method='chain-thaw')


        print(self.__model.summary())


    def getEncoding(self, listOfSentences):
        texts_unicode = [s.decode('utf-8') for s in listOfSentences]
        tokenized, _, _ = self.__st.tokenize_sentences(texts_unicode)
        #print('Encoding texts..')
        encoding = self.__model.predict(tokenized)
        return encoding





import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(dataPath+'Emobank.csv', sep='\t', quoting=csv.QUOTE_NONE, error_bad_lines=False)
content = df.iloc[:,1].tolist()

X = []
for i in range(0,len(df)):
    line = df.iloc[i][1]
    X.append(line)


doms = []
for i in range(0,len(df)):
    doms.append(df.iloc[i][4])


y = np.zeros((len(doms),1))

for i in range(0,len(doms)):
    y[i][0] = doms[i]

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


print y_train.shape


#encodingModel = DeepmojiEncoding(X_train,y_train, numEpoch = 5)


#train_encoding = encodingModel.getEncoding(np.array(X_train))
#test_encoding = encodingModel.getEncoding(np.array(X_test))

#np.save(dataPath+'Emobank_deepmoji_transfer_encoding_dominance_train.csv', train_encoding)
#np.save(dataPath+'Emobank_deepmoji_transfer_encoding_dominance_test.csv', test_encoding)


y_truth = np.load(dataPath+'Emobank_deepmoji_transfer_encoding_dominance_test.csv.npy').tolist()
y_truth = [x[0] for x in y_truth]
y_test = [x[0] for x in y_test]


import scipy

print 'Corr:', scipy.stats.pearsonr(y_test,y_truth)
