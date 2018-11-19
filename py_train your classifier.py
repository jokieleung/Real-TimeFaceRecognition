
# coding: utf-8

# # This script processing images and training your own  face classifier.

# In[1]:

import tensorflow as tf
import numpy as np
import cv2

import os
from os.path import join as pjoin
# import sys
# import copy
# import detect_face
# import nn4 as network
# import matplotlib.pyplot as plt
#
#
# import sklearn
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics  
from sklearn.externals import joblib
# from sklearn.svm import SVC
import net

# get_ipython().run_line_magic('pylab', 'inline')
###### train_dir containing one subdirectory per image class
#should like this:
#-->train_dir:
#     --->pic_me:
#            me1.jpg
#            me2.jpg
#            ...
#     --->pic_others:
#           other1.jpg
#            other2.jpg
#            ...
data_dir='./xiaoyu_classifier_train_dir/'#your own train folder

# In[2]:


#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

#facenet embedding parameters

model_dir='./model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=160 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."



# In[3]:


#建立人脸检测模型，加载参数
print('Creating networks and loading parameters')
gpu_memory_fraction=1.0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = net.create_mtcnn(sess, './model_check_point/')

# In[4]:

#Jokie way to build new embedding model

tf.reset_default_graph()
sess = tf.Session()
print('建立facenet embedding模型')
saver = tf.train.import_meta_graph('./model_check_point/model-20180408-102900.meta')
saver.restore(sess, './model_check_point/model-20180408-102900.ckpt-90')
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

embedding_size = embeddings.get_shape()[1]
print('embedding_size: ',embedding_size)
print('facenet embedding模型建立完毕')

def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def read_img(person_dir,f):
    img=cv2.imread(pjoin(person_dir, f))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    if gray.ndim == 2:
        img = to_rgb(gray)
    return img

def load_data(data_dir):
    data = {}
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)
        
        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]

        data[guy] = curr_pics

    return data

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y
# In[7]:


data=load_data(data_dir)
print(data.keys())
guys=[]
# for key in data.iterkeys():
for key in data.keys():
    guys.append(key)
    print('foler:{},image numbers：{}'.format(key,len(data[key])))

# In[ ]:

train_x=[]
train_y=[]
# 原来的keys=['other','video_guai','video_me']
# jokie guys = ['pic_others', 'pic_qian','pic_XiaoYu', 'pic_zeng']
guys_dic={'pic_XiaoYu':0,'pic_qian':1,'pic_zeng':2,'pic_others':3}

for guy in guys:
    print(guy)
    for x in data[guy]:
        bounding_boxes, _ = net.detect_face(x, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]#number of faces

        for face_position in bounding_boxes:
            try:
                face_position=face_position.astype(int)
                #print(face_position[0:4])
                cv2.rectangle(x, (face_position[0], face_position[1]), (face_position[2], face_position[3]), (0, 255, 0), 2)
                crop=x[face_position[1]:face_position[3],
                     face_position[0]:face_position[2],]

                crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC )
                crop_data=crop.reshape(-1,160,160,3)
                prewhitened = prewhiten(crop_data)

                emb_data = sess.run([embeddings],
                                    feed_dict={images_placeholder: np.array(prewhitened), phase_train_placeholder: False })[0]


                train_x.append(emb_data)
                guy_type_label = guys_dic[guy]
                train_y.append(guy_type_label)
                print('%s \'s label is %d' %(guy,guy_type_label))
            except:
                pass
    print('num of train_x',len(train_x))

print('embedding搞完了，样本数为：{}'.format(len(train_x)))


# In[11]:


#train/test split
train_x=np.array(train_x)
train_x=train_x.reshape(-1,512)
train_y=np.array(train_y)
print(train_x.shape)
print(train_y.shape)


X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.1, random_state=42)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# # KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=11)
    print('Training the KNN classifier...')
    model.fit(train_x, train_y)
    print('Training the completed.')
    return model

classifiers = knn_classifier

model = classifiers(X_train,y_train)
predict = model.predict(X_test)

accuracy = metrics.accuracy_score(y_test, predict)
print ('accuracy: %.2f%%' % (100 * accuracy)  )

#save model
joblib.dump(model, './model_check_point/k11N_xy_classifier_4type.model')

# In[13]:

model = joblib.load('./model_check_point/k11N_xy_classifier_4type.model')
predict = model.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predict)
print ('accuracy: %.2f%%' % (100 * accuracy)  )




# In[12]:
# svc Classifier
# def svc_classifier(train_x, train_y):
#     # Train classifier
#
#     model = SVC(kernel='poly', degree=10, gamma=1, coef0=0, probability=True)  # accuracy:
#     print('Training svc classifier...')
#     model.fit(train_x, train_y)
#     print('Training completed.')
#     return model
#
# classifiers = svc_classifier
#
# model = classifiers(X_train, y_train)
# predict = model.predict(X_test)
#
# accuracy = metrics.accuracy_score(y_test, predict)
# print('accuracy: %.2f%%' % (100 * accuracy))
#
# # save model
# joblib.dump(model, './model_check_point/xiaoyu_svc_classifier.model')
#
# # In[13]:
#
# model = joblib.load('./model_check_point/xiaoyu_svc_classifier.model')
# predict = model.predict(X_test)
# accuracy = metrics.accuracy_score(y_test, predict)
# print('loaded svc model accuracy: %.2f%%' % (100 * accuracy))
