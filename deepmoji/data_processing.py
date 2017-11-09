import numpy as  np
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH, NB_TOKENS

dataPath = '/home/royal/data/PersonalAttack/'
train_x = np.load(dataPath+'train_x_personal_attack.npy')
train_y = np.load(dataPath+'train_y_personal_attack.npy')
dev_x = np.load(dataPath+'dev_x_personal_attack.npy')
dev_y = np.load(dataPath+'dev_y_personal_attack.npy')
test_x = np.load(dataPath+'test_x_personal_attack.npy')
test_y = np.load(dataPath+'test_y_personal_attack.npy')


MAX_SENT_LENGTH = 60
MAX_SENTS = 5 

# In[9]:

from deepmoji.sentence_tokenizer import SentenceTokenizer
import json

vocabulary = {}
with open(VOCAB_PATH, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary,MAX_SENT_LENGTH)



def vectorize(doc):
    sentences = doc.split('NEWLINE_TOKEN')
    sentences = filter(lambda x: x.strip() != '', sentences)
    texts_unicode = [s.decode('utf-8') for s in sentences]
    texts_unicode = filter(lambda x: x.strip() != '', texts_unicode)
    try:
        tokenized, _, _ = st.tokenize_sentences(texts_unicode)
        return tokenized[:MAX_SENTS]
    except:
        return np.array([]).reshape((0,0))
    

def helperfunc(lis):
    mat_lis = np.zeros((len(lis), MAX_SENTS, MAX_SENT_LENGTH))
   
    for i in range(0,len(lis)):
        doc = lis[i]
        for j in range(0,len(doc)):
            sents = doc[j]
            for k in range(0, len(sents)):
                mat_lis[i][j][k] = lis[i][j][k]

    return mat_lis

n = 10000

    
train_new_x = np.array([vectorize(doc) for doc in train_x.tolist()])
dev_new_x = np.array([vectorize(doc) for doc in dev_x.tolist()])
test_new_x = np.array([vectorize(doc) for doc in test_x.tolist()])

train_new_y  = np.vectorize(lambda x: 1 if x else 0)(train_y)[:n]
dev_new_y  = np.vectorize(lambda x: 1 if x else 0)(dev_y)[:n]
test_new_y  = np.vectorize(lambda x: 1 if x else 0)(test_y)[:n]



train_x = helperfunc(train_new_x)
dev_x = helperfunc(dev_new_x)
test_x = helperfunc(test_new_x)


np.save(dataPath+'train_x_features',train_x)
np.save(dataPath+'dev_x_features',dev_x)
np.save(dataPath+'test_x_fetaures',test_x)


