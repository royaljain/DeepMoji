import os

#os.environ['KERAS_BACKEND']='theano'

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

def change_trainable(layer, trainable, verbose=False):
    layer.trainable = trainable
    if type(layer) == Bidirectional:
        layer.backward_layer.trainable = trainable
        layer.forward_layer.trainable = trainable
    if type(layer) == TimeDistributed:
        layer.backward_layer.trainable = trainable
    if verbose:
        action = 'Unfroze' if trainable else 'Froze'
        print("{} {}".format(action, layer.name))


def freeze(model):
    for l in model.layers:
        if len(l.trainable_weights):
            trainable = False
            change_trainable(l, trainable, verbose=True)

    return model


sentEncoder = freeze(sentEncoder)
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

dataPath = '/home/royal/data/PersonalAttack/'
train_y = np.load(dataPath+'train_y_personal_attack.npy')
dev_y = np.load(dataPath+'dev_y_personal_attack.npy')
test_y = np.load(dataPath+'test_y_personal_attack.npy')


train_new_y  = np.vectorize(lambda x: 1 if x else 0)(train_y)
dev_new_y  = np.vectorize(lambda x: 1 if x else 0)(dev_y)
test_new_y  = np.vectorize(lambda x: 1 if x else 0)(test_y)



train_x = np.load(dataPath+'train_x_features.npy')
dev_x = np.load(dataPath+'dev_x_features.npy')
test_x = np.load(dataPath+'test_x_features.npy')



callbacks = [EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto'),
             ModelCheckpoint(dataPath+'/weights.hdf5', monitor='val_loss', verbose=1)]

model.fit(train_x, train_new_y, validation_data=(dev_x, dev_new_y),
          epochs=1, batch_size=250, verbose=1, callbacks = callbacks)


y_pred =  model.predict(test_x)
np.save(dataPath + 'final_output', y_pred)


intermediate_model = Model(inputs=[model.input], outputs=[model.get_layer('attention_layer').output])


encoding = intermediate_model.predict(train_x)
np.save(dataPath + 'encoding_train', encoding)



encoding = intermediate_model.predict(dev_x)
np.save(dataPath + 'encoding_dev', encoding)


encoding = intermediate_model.predict(test_x)
np.save(dataPath + 'encoding_test', encoding)





