import sys
sys.path.insert(0,'/Users/royal/Desktop/MyDeepMoji/DeepMoji/')

import os

os.environ['KERAS_BACKEND']='theano'

from deepmoji.model_def import deepmoji_feature_encoding
from keras.layers.merge import Concatenate
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, SpatialDropout1D,     LSTM, Activation, TimeDistributed, GRU
from keras.models import Model
from deepmoji.global_variables import PRETRAINED_PATH, VOCAB_PATH, NB_TOKENS
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, Dropout, LSTM, GRU, Bidirectional, TimeDistributed

from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.callbacks import ModelCheckpoint, EarlyStopping

# In[42]:

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None



MAX_SENT_LENGTH = 60
MAX_SENTS = 5

sentEncoder = deepmoji_feature_encoding(MAX_SENT_LENGTH, weight_path=PRETRAINED_PATH)

print sentEncoder.summary()


# In[43]:


review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)

layer = TimeDistributed(sentEncoder)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = AttentionWeightedAverage(name='attention_layer')(l_dense_sent)
preds = Dense(1, activation='sigmoid')(l_att_sent)
model = Model(review_input, preds)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])



print model.summary()



import numpy as np

dataPath = '/Users/royal/Desktop/WikiAbuseDataset/PersonalAttack/'
train_x = np.load(dataPath+'train_x_personal_attack.npy')
train_y = np.load(dataPath+'train_y_personal_attack.npy')
dev_x = np.load(dataPath+'dev_x_personal_attack.npy')
dev_y = np.load(dataPath+'dev_y_personal_attack.npy')
test_x = np.load(dataPath+'test_x_personal_attack.npy')
test_y = np.load(dataPath+'test_y_personal_attack.npy')


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
    tokenized, _, _ = st.tokenize_sentences(texts_unicode)
    return tokenized[:MAX_SENTS]
    

def helperfunc(lis):
    mat_lis = np.zeros((len(lis), MAX_SENTS, MAX_SENT_LENGTH))
   
    for i in range(0,len(lis)):
        doc = lis[i]
        for j in range(0,len(doc)):
            sents = doc[j]
            for k in range(0, len(sents)):
                mat_lis[i][j][k] = lis[i][j][k]

    return mat_lis


    
train_new_x = np.array([vectorize(doc) for doc in train_x.tolist()[:2]])
dev_new_x = np.array([vectorize(doc) for doc in dev_x.tolist()[:2]])
test_new_x = np.array([vectorize(doc) for doc in test_x.tolist()[:2]])

train_new_y  = np.vectorize(lambda x: 1 if x else 0)(train_y)[:2]
dev_new_y  = np.vectorize(lambda x: 1 if x else 0)(dev_y)[:2]
test_new_y  = np.vectorize(lambda x: 1 if x else 0)(test_y)[:2]



train_x = helperfunc(train_new_x)
dev_x = helperfunc(dev_new_x)
test_x = helperfunc(test_new_x)



callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto'),
             ModelCheckpoint(dataPath+'/weights.hdf5', monitor='val_loss', verbose=1)]

model.fit(train_x, train_new_y, validation_data=(dev_x, dev_new_y),
          epochs=1, batch_size=1, verbose=2, callbacks = callbacks)


y_pred =  model.predict(test_x)
np.save(dataPath + 'final_output', y_pred)


intermediate_model = Model(inputs=[model.input], outputs=[model.get_layer('attention_layer').output])


encoding = intermediate_model.predict(train_x)
np.save(dataPath + 'encoding_train', encoding)



encoding = intermediate_model.predict(dev_x)
np.save(dataPath + 'encoding_dev', encoding)


encoding = intermediate_model.predict(test_x)
np.save(dataPath + 'encoding_test', encoding)





