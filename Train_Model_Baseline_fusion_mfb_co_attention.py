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
    

    
    
  

    
    model_img=Model(inputs=x_input,outputs=x)

    return model_img


# In[ ]:


def lstm_text_model(number_of_token,length_of_question):
    y_input=Input((length_of_question,))
    y=Embedding(number_of_token,100, input_length=length_of_question, trainable=True,mask_zero=True)(y_input)
    y=LSTM(units=1024, return_sequences=True,activation="tanh")(y)
    y=LSTM(units=1024, return_sequences=False,activation="tanh")(y)
    y=Reshape((length_of_question,1,1024))(y)
    

    model_txt=Model(inputs=y_input,outputs=y)

    return model_txt


# In[ ]:


def VQA_MFB(image_model,text_model):
    
#     lstm_out=lstm.output


#     '''
#     Question Attention
#     '''

    qatt_conv1=Convolution2D(512,(1,1))(text_model.output)
    qatt_relu = LeakyReLU()(qatt_conv1)
    qatt_conv2 = Convolution2D(2,(1,1)) (qatt_relu) # (N,L,1,2)
    qatt_conv2=Lambda(lambda x: K.squeeze(x, axis=2))(qatt_conv2)
    qatt_conv2 = Permute((2,1))(qatt_conv2)

    qatt_softmax = Activation("softmax")(qatt_conv2)
    qatt_softmax =Reshape( (11, 1,2))(qatt_softmax)


    def t_qatt_mask(tensors):
    #     lstm=lstm_text_model(emb_mat,emb_mat.shape[0],padding_train_ques.shape[1])
    #     lstm_out=lstm.output
        qatt_feature_list = []
        ten1=tensors[0]
        ten2=tensors[1]

        for i in range(2):
            t_qatt_mask = ten1[:,:,:,i]  # (N,1,L,1)
            t_qatt_mask=K.reshape(t_qatt_mask,(-1,11,1,1))
            t_qatt_mask = t_qatt_mask * ten2  # (N,1024,L,1)
    #         print(t_qatt_mask)
        #     t_qatt_mask = K.sum(t_qatt_mask,axis=1,keepdims=True)
            t_qatt_mask= K.sum(t_qatt_mask,axis=1,keepdims=True)
    #         print(t_qatt_mask)

            qatt_feature_list.append(t_qatt_mask)

        qatt_feature_concat = K.concatenate(qatt_feature_list)  # (N,2048,1,1)
        return qatt_feature_concat


    q_feat_resh=Lambda(t_qatt_mask)([qatt_softmax,text_model.output])

    q_feat_resh =Lambda(lambda x: K.squeeze(x, axis=2))(q_feat_resh)

    q_feat_resh=Permute((2,1))(q_feat_resh)
    q_feat_resh=Reshape([2048])(q_feat_resh)                                           # (N,2048)




#     '''
#      MFB Image  with Attention

#     '''

    # image_feature=vgg.output


    i_feat_resh = Reshape((196,1, 512))(image_model.output)  # (N,512,196)

    iatt_fc = Dense(1024,activation="tanh")(q_feat_resh)  # (N,5000)
    iatt_resh = Reshape( (1, 1,1024))(iatt_fc)  # (N,5000,1,1)
    iatt_conv = Convolution2D(1024,(1,1)) (i_feat_resh) # (N,5000,196,1)
    iatt_conv=LeakyReLU()(iatt_conv)
    iatt_eltwise = multiply([iatt_resh , iatt_conv])  # (N,5000,196,1)
    iatt_droped = Dropout(0.1)(iatt_eltwise)

    iatt_permute1 = Permute(( 3 ,2, 1))(iatt_droped)  # (N,196,5000,1)
    iatt_resh2 = Reshape( (512, 2,196))(iatt_permute1)
    iatt_sum =Lambda(lambda x: K.sum(x, axis=2,keepdims=True))(iatt_resh2)
    iatt_permute2 =Permute((3, 2, 1))(iatt_sum)  # (N,1000,196,1)
    iatt_sqrt = Lambda(lambda x:K.sqrt(Activation("relu")(x)) - K.sqrt(Activation("relu")(-x)))(iatt_permute2)
    iatt_sqrt =Reshape([-1])(iatt_sqrt)
    iatt_l2=Lambda(lambda x: K.l2_normalize(x,axis=1))(iatt_sqrt)
    iatt_l2=Reshape((196,1,512))(iatt_l2)


    iatt_conv1 =Convolution2D (512,(1,1)) (iatt_l2) # (N,512,196,1)
    iatt_relu = LeakyReLU()(iatt_conv1)
    iatt_conv2 = Convolution2D(2,(1,1))(iatt_relu)  # (N,2,196,1)
    iatt_conv2 = Reshape((2,196) )(iatt_conv2)
    iatt_softmax = Activation("softmax")(iatt_conv2)
    iatt_softmax = Reshape((196,1, 2))(iatt_softmax)


    def iatt_feature_list(tensors):
    #     global i_feat_resh
        ten3=tensors[0]
        ten4=tensors[1]
        iatt_feature_list = []
        for j in range(2):
            iatt_mask = ten3[:,:,:,j] # (N,1,196,1)
            iatt_mask=K.reshape(iatt_mask,(-1,196,1,1))   
            iatt_mask = iatt_mask * ten4  # (N,512,196,1)
    #         print(iatt_mask)
            iatt_mask = K.sum(iatt_mask,axis=1 ,keepdims=True)
            iatt_feature_list.append(iatt_mask)
        iatt_feature_cat = K.concatenate(iatt_feature_list)  # (N,1024,1,1)
        return iatt_feature_cat

    iatt_feature_cat=Lambda(iatt_feature_list)([iatt_softmax,i_feat_resh])
    iatt_feature_cat =Lambda(lambda x: K.squeeze(x, axis=2))(iatt_feature_cat)
    iatt_feature_cat=Permute((2,1))(iatt_feature_cat)
    iatt_feature_cat=Reshape([1024])(iatt_feature_cat)
    
    
#         '''
#     Fine-grained Image-Question MFH fusion
#     '''
    # print(q_feat_resh.shape)
    # print(bert_encode.shape)

    # if mode != 'train':
    #     q_feat_resh = q_feat_resh.unsqueeze(0)

    mfb_q = Dense(1024, activation="tanh")(q_feat_resh)  # (N,5000)
    mfb_i = Dense(1024)(iatt_feature_cat)  # (N,5000)
    mfb_i=LeakyReLU()(mfb_i)
    mfb_eltwise =multiply([mfb_q, mfb_i])
    mfb_drop1 = Dropout(0.1)(mfb_eltwise)
    mfb_resh = Reshape(  (512,2,1))(mfb_drop1)  # (N,1,1000,5)
    mfb_sum = Lambda(lambda x: K.sum(x, axis=2, keepdims=True))(mfb_resh)
    mfb_out = Reshape([512])(mfb_sum)
    mfb_sqrt =Lambda(lambda x: K.sqrt(Activation("relu")(x)) - K.sqrt(Activation("relu")(-x)))(mfb_out)

    # if mode != 'train':
    #     mfb_sqrt = mfb_sqrt.unsqueeze(0)

    mfb_l2_1 = Lambda(lambda x: K.l2_normalize(x,axis=1))(mfb_sqrt)

#     mfb_q2 = Dense(1024, activation="tanh")(text_model.output) # (N,5000)
#     mfb_i2 = Dense(1024)(iatt_feature_cat)  # (N,5000)
#     mfb_i2=LeakyReLU()(mfb_i2)
#     mfb_eltwise2 = multiply([mfb_q2, mfb_i2])  # (N,5000)
#     mfb_eltwise2 =multiply([mfb_eltwise2, mfb_drop1])
#     mfb_drop2 = Dropout(0.1)(mfb_eltwise2)
#     mfb_resh2 = Reshape( (512,2,1))(mfb_drop2)
#     #                              # (N,1,1000,5)
#     mfb_sum2 = Lambda(lambda x: K.sum(x, 2, keepdims=True))(mfb_resh2)
#     mfb_out2 =Reshape([512])(mfb_sum2)
#     mfb_sqrt2 =Lambda(lambda x: K.sqrt(Activation("relu")(x)) - K.sqrt(Activation("relu")(-x)))(mfb_out2)

#     # if mode != 'train':
#     #     mfb_sqrt2 = mfb_sqrt2.unsqueeze(0)
#     mfb_l2_2 =Lambda(lambda x: K.l2_normalize(x,axis=1))(mfb_sqrt2)

#     mfb_l2_3 = Concatenate(axis=1)([mfb_l2_1, mfb_l2_2])  # (N,2000)
    fc1=Dense(1024)(mfb_l2_1)
    fc1_lr=LeakyReLU()(fc1)

    prediction = Dense(len(class_to_label),activation="softmax")(fc1_lr)
    
    vqa_model=Model(inputs=[image_model.input,text_model.input],outputs=prediction)
    vqa_model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=["accuracy"])
    
    return vqa_model
    


# In[ ]:


model_vqa=VQA_MFB(vgg_image_model(),lstm_text_model(len(word_to_index),12))


# In[ ]:


callback = [EarlyStopping(monitor='val_loss',mode="min",verbose=1, patience=30),
             ModelCheckpoint('train_{}_{}.h5'.format("vgg","lstm_baseline_mfb"), monitor='val_loss',mode="min" ,verbose=1,save_best_only=True)]
model_fit=model_vqa.fit_generator(training_data,steps_per_epoch=len(training_data_samples)//32,epochs=300,validation_data=valid_data,validation_steps=len(valid_data_samples)//32,verbose=1, callbacks=callback)


# In[ ]:


model=load_model("train_vgg_lstm_baseline_mfb.h5")


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



