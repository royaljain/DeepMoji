import json

from deepmoji.model_def import deepmoji_transfer, load_specific_weights
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH, NB_TOKENS
from deepmoji.class_avg_finetuning import class_avg_finetune
from deepmoji.finetuning import calculate_batchsize_maxlen
from deepmoji.sentence_tokenizer import SentenceTokenizer
import numpy as np
from keras.utils import to_categorical



import os

absPath = '/Users/royal/Desktop/MyDeepMoji/DeepMoji/'
relPath = 'data/WikiPersonnalAttack/'

dataPath = os.path.join(absPath, relPath)


print 'DATAPATH : ', dataPath

class DeepmojiEncoding:

    def __init__(self):

        print('Tokenizing using dictionary from {}'.format(VOCAB_PATH))


        train_x = np.load(dataPath + 'train_x_personal_attack.npy')
        dev_x = np.load(dataPath + 'dev_x_personal_attack.npy')
        test_x = np.load(dataPath + 'test_x_personal_attack.npy')

        train_y = np.load(dataPath + 'train_y_personal_attack.npy')
        dev_y = np.load(dataPath + 'dev_y_personal_attack.npy')
        test_y = np.load(dataPath + 'test_y_personal_attack.npy')

        train_y = to_categorical(train_y)
        dev_y = to_categorical(dev_y)
        test_y = to_categorical(test_y)


        print len(train_x)
        print len(dev_x)
        print len(test_x)
        print len(train_y)
        print len(dev_y)
        print len(test_y)


        print train_x[0]
        print train_y[0]
        print len(train_y[0])
        print len(dev_y[0])
        print len(train_y[0])

        vocabulary = {}
        with open(VOCAB_PATH, 'r') as f:
            vocabulary = json.load(f)

        all_x = np.concatenate((train_x, dev_x, test_x))

        batchsize,  maxlen = calculate_batchsize_maxlen(all_x)

        print 'MaxLen: ', maxlen
        print 'BatchSize: ', batchsize

        self.__st = SentenceTokenizer(vocabulary, maxlen)

        print('Loading model from {}.'.format(PRETRAINED_PATH))
        self.__model = deepmoji_transfer(nb_classes=2, maxlen=maxlen, weight_path=PRETRAINED_PATH )


        train_texts_unicode = [s.decode('utf-8') for s in train_x]
        train_tokenized, _, _ = self.__st.tokenize_sentences(train_texts_unicode)

        dev_texts_unicode = [s.decode('utf-8') for s in dev_x]
        dev_tokenized, _, _ = self.__st.tokenize_sentences(dev_texts_unicode)

        test_texts_unicode = [s.decode('utf-8') for s in test_x]
        test_tokenized, _, _ = self.__st.tokenize_sentences(test_texts_unicode)


        data_texts = [train_x, dev_x, test_x]
        data_labels = [train_y, dev_y, test_y]


        self.__model, f1 = class_avg_finetune(self.__model, data_texts, data_labels,
                                       nb_classes=2, nb_epochs=1, batch_size= batchsize, method='chain-thaw')


        print(self.__model.summary())


        encoding = self.__model.predict(train_tokenized)
        np.save(dataPath + 'finetune_encoding_train_x', encoding)

        encoding = self.__model.predict(dev_tokenized)
        np.save(dataPath + 'finetune_encoding_dev_x', encoding)

        encoding = self.__model.predict(test_tokenized)
        np.save(dataPath + 'finetune_encoding_test_x', encoding)




DeepmojiEncoding()



