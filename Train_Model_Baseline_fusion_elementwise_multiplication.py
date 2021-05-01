#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import tensorflow as tf
from tensorflow import keras
from keras import backend as K

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image

from keras.models import Sequential,Model,load_model
from keras.layers import LeakyReLU,Input,Reshape, Lambda, Dense,Permute,multiply, Activation, Dropout,Multiply, LSTM, Flatten, Embedding, merge,TimeDistributed,Concatenate,Bidirectional,BatchNormalization,Add,AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import metrics
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import plot_model
from keras.utils import np_utils
# import np_utils
import random

# from keras_bert import get_custom_objects
# from tqdm import tqdm
# from chardet import detect
# import keras
# # from keras_bert import load_trained_model_from_checkpoint
# # from keras_bert import Tokenizer as k_Tokenizer

import sklearn
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,LabelBinarizer
# from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score,roc_curve,confusion_matrix,auc
from sklearn.utils import class_weight
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import nltk
import json
import copy
import os, argparse,cv2,h5py,string
import collections
from collections import Counter


from scipy import interp
from itertools import cycle
import seaborn as sns

# from deepexplain.tensorflow import DeepExplain
# import codecs

# import innvestigate
# import innvestigate.utils as iutils

# from skimage.measure import compare_ssim as ssim
# from skimage import data
# from skimage.color import rgb2gray

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# from tensorflow.compat.v1 import ConfigProto,InteractiveSession
# from keras.backend.tensorflow_backend import set_session,clear_session,get_session



# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 1
# set_session(tf.Session(config=config))

# config = tf.ConfigProto()
# config.gpu_options.visible_device_list = "1"
# config.gpu_options.per_process_gpu_memory_fraction = .3
# set_session(tf.Session(config=config))

# G =tf.Graph()
# sess1 = tf.Session(graph=G, config=tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list='0')))
# sess2 = tf.Session(graph=G, config=tf.ConfigProto(log_device_placement=False,gpu_options=tf.GPUOptions(allow_growth=True,visible_device_list='1')))



# Seed value

# Apparently you may use different seed values at each stage
seed_value= 443512

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.set_random_seed(seed_value)


# In[ ]:


os.getcwd()


# In[ ]:


root_dir="./Michael_VQA"
img_dir="./Original_Image"
img_dir_write="./Original_Image_Final"


# In[ ]:


def image_write(image_path_extract,image_path_write,image_list):

    for item in range(len(os.listdir(img_dir))):
        find_img=cv2.imread(os.path.join(img_dir,img_list[item]))
        img_resize=cv2.resize(find_img,(224,224))
    #     final_img=img_resize/255
        cv2.imwrite(os.path.join(img_dir_write ,img_list[item]), img_resize)
        

    return print("image write is complete")


# In[ ]:


image_write(img_dir,img_dir_write,image_list,os.listdir(img_dir))


# In[ ]:


class_to_label=json.load(open(root_dir+"\\harvey_class_to_label.json"))
label_to_class=json.load(open(root_dir+"\\harvey_label_to_class.json"))


# In[ ]:


word_to_index=json.load(open(root_dir+"\\harvey_question_word_to_token.json"))
index_to_word=json.load(open(root_dir+"\\harvey_question_token_to_word.json"))


# In[ ]:


training_data_dic=json.load(open(root_dir+"\\Generated_training_question_harvey_VQA.json"))
valid_data_dic=json.load(open(root_dir+"\\Generated_valid_question_harvey_VQA.json"))
test_data_dic=json.load(open(root_dir+"\\Generated_test_question_harvey_VQA.json"))


# In[ ]:


training_data_dic


# In[ ]:


def dictionary_preprocess(dictionary_input):
    image_id,question,answer,samples_list,question_type=[],[],[],[],[]
    
    for item in range(len(dictionary_input)):
        image_id.append(dictionary_input[str(item)]["Image_ID"])
        question.append(dictionary_input[str(item)]["Question"])
        answer.append(dictionary_input[str(item)]["Ground_Truth"])
        question_type.append(dictionary_input[str(item)]["Question_Type"])
        
        
        
    for i,q,a,qt in  zip(image_id,question,answer,question_type):
        samples_list.append([i,q,a,qt])
        
    return samples_list
        
        
    
    
    


# In[ ]:


training_data_samples=dictionary_preprocess(training_data_dic)
valid_data_samples=dictionary_preprocess(valid_data_dic)
test_data_samples=dictionary_preprocess(test_data_dic)


# In[ ]:


tt=[training_data_samples[i][3] for i in range(len(training_data_samples)) if training_data_samples[i][3]== "Entire_Image_condition"]


# In[ ]:


def question_preprocess(ques_input,max_length_ques):
 

        inp_ques=(" ".join(ques_input.split()).lower().translate(str.maketrans('', '', string.punctuation))) 
        ques_seq=[word_to_index[word] for word in inp_ques.split(" ")]

        if len(ques_seq)!=max_length_ques:
            ques_copy=copy.copy(ques_seq)
            for e in range(max_length_ques-len(ques_copy)):
                ques_copy.append(0)
            return ques_copy
        else:
            return ques_seq
#         pad_ques=pad_sequences(ques,maxlen=self.length_ques,padding="post")


# In[ ]:


def label_preprocess(input_answer):
#     input_answer=
    
    ans_inp=class_to_label[input_answer.lower()]
    
    return(np_utils.to_categorical(ans_inp, num_classes=len(class_to_label)))
    
    


# In[ ]:


def shuffle_data(data):
    
    return random.shuffle(data)
    
#     return data


# In[ ]:


def data_generator(samples_list,batch_size=32,shuffle_arg=True,dim=224,n_channel=3,length_ques=11,num_class=len(class_to_label)):
    
    
    num_samples=len(samples_list)
#     print(num_samples)
    while True:
        shuffle_data(samples_list)
#         print(samples_list)
        
        for offset in range(0,num_samples,batch_size):
#             print(offset)
            batch_samples=samples_list[offset:offset+batch_size]
#             print(batch_samples)
            X1 = np.zeros((batch_size, dim,dim, n_channel),dtype='float32')

            X2=np.zeros((batch_size, length_ques),dtype='float32')
            y=np.zeros((batch_size, num_class),dtype='float32')
            
            for batch_sample in range(len(batch_samples)):
                img_name=batch_samples[batch_sample][0]
#                 print(img_name)
                ques_provide=batch_samples[batch_sample][1]
                label_provide=str(batch_samples[batch_sample][2])
#                 print(type(label_provide))
                
                X1[batch_sample,] = cv2.imread(os.path.join(img_dir_write,img_name))/255
                X2[batch_sample,]=question_preprocess(ques_provide,11)
                y[batch_sample,]=label_preprocess(label_provide)
                
                
                
            yield [X1,X2],y
                
                
        


# In[ ]:


training_data=data_generator(samples_list=training_data_samples,batch_size=32,shuffle_arg=True,dim=224,n_channel=3,length_ques=11,num_class=len(class_to_label))
valid_data=data_generator(samples_list=valid_data_samples,batch_size=32,shuffle_arg=True,dim=224,n_channel=3,length_ques=11,num_class=len(class_to_label))


# In[ ]:


def vgg_image_model():
    
    x_input=Input((224,224,3))
    x=ZeroPadding2D((1,1))(x_input)
    x=Convolution2D(64, (3, 3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
    
    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(64,(3,3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
    x=MaxPooling2D((2,2), strides=(2,2))(x)
    
    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(128, (3, 3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
    
    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(128, (3, 3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
    x=MaxPooling2D((2,2), strides=(2,2))(x)
    
    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(256, (3, 3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
    
    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(256,(3,3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
   
    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(256,(3,3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
    x=MaxPooling2D((2,2), strides=(2,2))(x)
    
    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(512, (3, 3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
    
    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(512, (3, 3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
    
    x=ZeroPadding2D((1,1))(x)
    x=Convolution2D(512,(3,3))(x)
    x=BatchNormalization(axis=3)(x)
    x=LeakyReLU()(x)
    x=MaxPooling2D((2,2), strides=(2,2))(x)
    x=Dense(1024)(x)
    x=LeakyReLU()(x)
    

    
    
    x=Flatten()(x)
    

    
    model_img=Model(inputs=x_input,outputs=x)

    return model_img


# In[ ]:


def lstm_text_model(number_of_token,length_of_question):
    y_input=Input((length_of_question,))
    y=Embedding(number_of_token,100, input_length=length_of_question, trainable=True,mask_zero=True)(y_input)
    y=LSTM(units=1024, return_sequences=True,activation="tanh")(y)
    y=LSTM(units=1024, return_sequences=False,activation="tanh")(y)

    

    model_txt=Model(inputs=y_input,outputs=y)

    return model_txt


# In[ ]:








vgg=vgg_image_model()
lstm=lstm_text_model(len(word_to_index),12)

combined = Multiply()([lstm.output,vgg.output])
c1=Dense(2048)(combined)
c2=LeakyReLU()(c1)
c3=Dense(1024)(c2)
c4=LeakyReLU()(c3)
c5=Dense(len(class_to_label))(c4)
c6=Activation("softmax")(c5)


model_vqa = Model(inputs=[vgg.input,lstm.input], outputs=c6)


# In[ ]:


model_vqa.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=["accuracy"])


# In[ ]:


callback = [EarlyStopping(monitor='val_loss',mode="min",verbose=1, patience=30),
             ModelCheckpoint('train_{}_{}.h5'.format("vgg","lstm_baseline_mul"), monitor='val_loss',mode="min" ,verbose=1,save_best_only=True)]
model_fit=model_vqa.fit_generator(training_data,steps_per_epoch=len(training_data_samples)//32,epochs=300,validation_data=valid_data,validation_steps=len(valid_data_samples)//32,verbose=1, callbacks=callback)


# In[ ]:


model=load_model("train_vgg_lstm_baseline_mul.h5")


# In[ ]:


def data_model_evaluate(data_sample,category=None):
    
    if category is not None:
    
        category_image_id=[data_sample[item_ind][0] for item_ind in range(len(data_sample)) if data_sample[item_ind][3]==category]
        category_ques=[data_sample[item_ind][1] for item_ind in range(len(data_sample)) if data_sample[item_ind][3]==category]
        category_gt=[data_sample[item_ind][2] for item_ind in range(len(data_sample)) if data_sample[item_ind][3]==category]

        if (len(category_image_id)==len(category_ques)) & (len(category_image_id)==len(category_gt)) & (len(category_gt)==len(category_ques)):

                eval_image = np.zeros((len(category_image_id), 224,224, 3),dtype='float32')

                eval_ques=np.zeros((len(category_image_id), 11),dtype='float32')
                eval_gt=np.zeros((len(category_image_id), len(class_to_label)),dtype='float32')

                for r in range(len(category_image_id)):
                    eval_image[r,] = cv2.imread(os.path.join(img_dir_write,category_image_id[r]))/255
                    eval_ques[r,]=question_preprocess(category_ques[r],11)
                    eval_gt[r,]=label_preprocess(str(category_gt[r]))


    else:
        category_image_id=[data_sample[item_ind][0] for item_ind in range(len(data_sample))]
        category_ques=[data_sample[item_ind][1] for item_ind in range(len(data_sample)) ]
        category_gt=[data_sample[item_ind][2] for item_ind in range(len(data_sample))]

        if (len(category_image_id)==len(category_ques)) & (len(category_image_id)==len(category_gt)) & (len(category_gt)==len(category_ques)):

                eval_image = np.zeros((len(category_image_id), 224,224, 3),dtype='float32')

                eval_ques=np.zeros((len(category_image_id), 11),dtype='float32')
                eval_gt=np.zeros((len(category_image_id), len(class_to_label)),dtype='float32')

                for r in range(len(category_image_id)):
                    eval_image[r,] = cv2.imread(os.path.join(img_dir_write,category_image_id[r]))/255
                    eval_ques[r,]=question_preprocess(category_ques[r],11)
                    eval_gt[r,]=label_preprocess(str(category_gt[r]))
        
    
    return [eval_image,eval_ques],eval_gt
                
    
    
    
    


# In[ ]:


dat=test_data_samples
all_=data_model_evaluate(dat)
cr=data_model_evaluate(dat,"Condition_Recognition")
sc=data_model_evaluate(dat,"Simple_Counting")
cc=data_model_evaluate(dat,"Complex_Counting")
bi=data_model_evaluate(dat,"Yes_No")
# rcr=data_model_evaluate(test_data_samples,"Road_Condition_Recognition")


# In[ ]:



print("accuracy for overall:", model.evaluate(all_[0],all_[1]))



print("accuracy for Simple_Counting:", model.evaluate(sc[0],sc[1]))

print("accuracy for Complex_Counting:", model.evaluate(cc[0],cc[1]))

print("accuracy for Yes_No:", model.evaluate(bi[0],bi[1]))
print("accuracy for Condition_Recognition:", model.evaluate(cr[0],cr[1]))
# print("accuracy for Road_Condition_Recognition:", model.evaluate(rcr[0],rcr[1]))



