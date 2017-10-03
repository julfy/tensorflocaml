from __future__ import division, print_function
import numpy as np
import tensorflow as tf

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Embedding, BatchNormalization, Dropout, GlobalMaxPooling1D, LSTM, GRU, Conv1D
from keras.regularizers import l2
from keras.models import load_model
#----------------------------------------------
from keras import backend as K
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
def save_model_as_graph(model,dir,name):
    sess = K.get_session()
    if K.learning_phase() != 0:
        raise Exception("Learning phase should have been set 0 before loading the model :/")
    model_output_nodes = [n.name.rsplit(':', 1)[0] for n in model.outputs]
    print ("Graph input nodes:",[n.name.rsplit(':', 1)[0] for n in model.inputs])
    print ("Graph output nodes:", model_output_nodes)
    # pred = [None]*num_output
    # pred_node_names = [None]*num_output
    # for i in range(num_output):
    #     pred_node_names[i] = output_node_names_of_final_network+str(i)
    #     pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), model_output_nodes)
    graph_io.write_graph(constant_graph, dir, name+".pb", as_text=False)
    graph_io.write_graph(constant_graph, dir, name+".pbtxt", as_text=True)

def save_model(model,dir,name):
    model.save(dir+"/"+name+".h5")

home = "."
data = home + "/data"
models = home + '/models'
# if not os.path.exists(models): os.mkdir(models)

import csv
# import random
def load_preprocessed(file):
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile,quoting=csv.QUOTE_NONNUMERIC)
        l = list(reader)
        # random.shuffle(l)
        x = []
        y = []
        for r in l:
            x.insert(0,r[:-1])
            y.insert(0,[r[-1]])
        return x,y

def slice2(l1,l2,l,r):
    num = len(l1)
    return l1[int(num*l):int(num*r)],l2[int(num*l):int(num*r)]

batch_size=300
maxlen = 1500 # max(list(map(len,x_train))) : 2383

def make_model(max_id):
    i = Input((maxlen,))
    x = Embedding(max_id, 50, mask_zero=True)(i)
    x = BatchNormalization()(x)
    x = Dropout(0.6)(x)
    x = LSTM(10, dropout=0.8, recurrent_dropout=0.8, recurrent_regularizer=l2(1.0),kernel_regularizer=l2(1.0))(x)
    x = BatchNormalization()(x)
    o = Dense(1, activation='sigmoid')(x)
    model = Model(inputs = i, outputs = o)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_weights(model,dir,name):
    try:
        model.load_weights("{}/{}".format(models,name))
        print ("Loaded:",name)
    except BaseException as e:
        print ("ERROR: failed to load from",name,":", str(e))

def convert (inp,out): # just convert the model
    K.set_learning_phase(0)
    model = load_model (inp)
    save_model_as_graph(model,models,out)
    del model

def train():
    K.set_learning_phase(1)

    x,y = load_preprocessed('alldump.csv')

    train_p = 0.75
    val_p = 0.1
    max_id = int(max(map(lambda e: max(e),x)))+1

    x_train,y_train = slice2(x,y,0,train_p)
    x_val,y_val = slice2(x,y,train_p,train_p+val_p)
    x_test,y_test = slice2(x,y,train_p+val_p,1)

    # x_train = x_train[:3000]
    # y_train = y_train[:3000]

    # x_train = np.expand_dims(x_train, axis=2) # Conv1D :/
    # x_test = np.expand_dims(x_test, axis=2) # Conv1D :/

    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)
    print('x_test shape:', x_test.shape)
    print('max_id:', max_id)

    model = make_model(max_id)
    # model = load_model ('')

    print (model.summary())

    model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_val, y_val))

    print('Evaluate...')
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('\nTest score:', score)
    print('Test accuracy:', acc)
    name = "{}/{:.4f}.h5".format(models,acc)
    model.save(name)
    print("Saved to:", name)
    del model

convert("models/0.8621.h5","final_") # note the input and output nodes those are required to use the model
# train ()
