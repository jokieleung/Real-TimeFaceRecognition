# coding: utf-8

import tensorflow as tf
import numpy as np
import cv2
from scipy import misc
from os.path import join as pjoin
import os
import net
import traceback
# import nn4 as network
# import random
# import sklearn

from sklearn.externals import joblib

# CUDA_VISIBLE_DEVICES=1 #Only device 1 will be seen

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

dataSetRootDir = './FaceDataSet/'

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

frame_interval=1 # frame intervals

# In[2]:一些用到的函数

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def Pic2Emb(PicName):
    # draw = cv2.imread(PicName)
    # gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
    #
    # if gray.ndim == 2:
    #     img = to_rgb(gray)

    # img = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    img = misc.imread(os.path.expanduser(PicName), mode='RGB')

    bounding_boxes, points = net.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    for face_position in bounding_boxes:
        face_position = face_position.astype(int)

        crop=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
        crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC )

        data=crop.reshape(-1,160,160,3)

        prewhitened = prewhiten(data)

        emb_data = sess.run(embeddings,
                            feed_dict={images_placeholder: np.array(prewhitened),
                                       phase_train_placeholder: False })

        print(PicName, emb_data.shape)

    return emb_data

def compare_pic(feature1,feature2):
    dist = np.sqrt(np.sum(np.square(np.subtract(feature1, feature2))))
    return dist

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret


def read_img(person_dir, f):
    img = cv2.imread(pjoin(person_dir, f))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if gray.ndim == 2:
        img = to_rgb(gray)
    return img


def load_raw_data(data_dir):
    raw_data = {}
    for guy in os.listdir(data_dir):
        person_dir = pjoin(data_dir, guy)

        curr_pics = [read_img(person_dir, f) for f in os.listdir(person_dir)]

        raw_data[guy] = curr_pics

    return raw_data

def GetLabel(X2get,dataSet_X,dataSet_Y,Dic_Label2Name):
    minDistance = 1.1
    for emb in dataSet_X:
        L2Distance = compare_pic(X2get,emb)
        if L2Distance < minDistance:
            minDistance = L2Distance

            #salution 1
            # index = dataSet_X.index(emb) # you dian wen ti

            # salution 2
            tim = 0
            for data in dataSet_X:
                # print((data == emb).all())
                if (data == emb).all():
                    index = tim
                tim += 1

            label = dataSet_Y[index]
    PersonName = Dic_Label2Name[label]
    return label, PersonName, minDistance

# In[3]:
#restore mtcnn model

print('Creating mtcnn networks and loading parameters')
gpu_memory_fraction=0.94
tf.reset_default_graph()

with tf.Graph().as_default() as g:
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction, allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False, allow_soft_placement=True), graph=g)
    with sess.as_default():
        pnet, rnet, onet = net.create_mtcnn(sess, './model_check_point/')
print('MTCNN networks created')

# In[4]:
# JoKie Restore facenet @20180912

# pretrained model by facenet(not fine tune)

# tf.reset_default_graph()
# sess = tf.Session()
# print('建立facenet embedding模型')
# saver = tf.train.import_meta_graph('./model_check_point/model-20180408-102900.meta')
# saver.restore(sess, './model_check_point/model-20180408-102900.ckpt-90')
# images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
# embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
# phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")


# fine tune model echo step 250k (batch==90)times    model-20181116-221515.ckpt-250452.data-00000-of-00001

tf.reset_default_graph()
sess = tf.Session()
print('建立facenet embedding模型')
saver = tf.train.import_meta_graph('./Latest_Fine_Tuning_model_20181118/model-20181116-221515.meta')
saver.restore(sess, './Latest_Fine_Tuning_model_20181118/model-20181116-221515.ckpt-250452')
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
print('fine tune model loaded')

print('facenet embedding模型建立完毕')

# In[5]:

# 准备数据库的人脸embedding
# Salution 1

raw_data=load_raw_data(dataSetRootDir)
print(raw_data.keys())

Person2LabelDic={}
Label2PersonDic={}
persons=[]
label_value=0
for person in raw_data.keys():
    persons.append(person)
    Person2LabelDic[person] = label_value
    Label2PersonDic[label_value] = person
    label_value += 1
    print('foler:{},image numbers：{}'.format(person, len(raw_data[person])))
print('Person2LabelDic:',Person2LabelDic)
print('Label2PersonDic:',Label2PersonDic)

train_x=[]
train_y=[]

for person in persons:
    print(person)
    for x in raw_data[person]:
        bounding_boxes, _ = net.detect_face(x, minsize, pnet, rnet, onet, threshold, factor)
        nrof_faces = bounding_boxes.shape[0]#number of faces
        print(bounding_boxes)
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
                guy_type_label = Person2LabelDic[person]
                train_y.append(guy_type_label)
                print('%s \'s label is %d' %(person,guy_type_label))
            except:
                pass
    print('num of train_x',len(train_x))

print('数据库embedding准备完毕，样本数为：{}'.format(len(train_x)))
print('length of train_y : ',len(train_y))
# print(train_x[:2])
print('shape of train_x',np.array(train_x).shape)
print('shape of train_y',np.array(train_y).shape)
print(train_y[:-1])

train_x = np.array(train_x).reshape((-1,512))

# # KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=5)
    print('Training the KNN classifier...')
    model.fit(train_x, train_y)
    print('Training the completed.')
    return model


classifiers = knn_classifier

model = classifiers(train_x,train_y)

# Salution 2 database's faces have been aligned
# raw_data=load_raw_data(dataSetRootDir)
# print(raw_data.keys())
#
# Person2LabelDic={}
# Label2PersonDic={}
# persons=[]
# label_value=0
# for person in raw_data.keys():
#     persons.append(person)
#     Person2LabelDic[person] = label_value
#     Label2PersonDic[label_value] = person
#     label_value += 1
#     print('foler:{},image numbers：{}'.format(person, len(raw_data[person])))
# print('Person2LabelDic:',Person2LabelDic)
# print('Label2PersonDic:',Label2PersonDic)
#
# train_x=[]
# train_y=[]
#
# for person in persons:
#     print(person)
#     for x in raw_data[person]:
#         #x is database's face, no need to align by mtcnn
#
#         try:
#             print('x shape',x.shape)
#             crop_data=x.reshape(-1,160,160,3)
#             print('crop data shape',crop_data.shape)
#             prewhitened = prewhiten(crop_data)
#
#             emb_data = sess.run([embeddings],
#                                 feed_dict={images_placeholder: np.array(prewhitened), phase_train_placeholder: False })[0]
#
#             train_x.append(emb_data)
#             guy_type_label = Person2LabelDic[person]
#             train_y.append(guy_type_label)
#             print('%s \'s label is %d' %(person,guy_type_label))
#         except:
#             pass
#     print('num of train_x',len(train_x))
#
# print('数据库embedding准备完毕，样本数为：{}'.format(len(train_x)))
# # print(train_x[:2])
# print(train_y[:-1])


# In[6]:
# 实时人脸识别与数据库进行对比，选取欧式距离最小的label，看距离是否小于1.1，若是则返回该label，否则输出unknown
# real time face detection and recognition

#obtaining frames from camera--->converting to gray--->converting to rgb
#--->detecting faces---->croping faces--->embedding--->classifying--->print

# video_capture.release()
# cv2.destroyAllWindows()

video_capture = cv2.VideoCapture(-1)
c=0

while True:
    # Capture frame-by-frame

    ret, frame = video_capture.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(frame.shape)
    
    timeF = frame_interval
    
    
    if(c%timeF == 0): # face detection every ‘frame_interval’ frames
        
        find_results=[]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if gray.ndim == 2:
            img = to_rgb(gray)

        bounding_boxes, _ = net.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

        # nrof_faces = bounding_boxes.shape[0]#number of faces
        #print('找到人脸数目为：{}'.format(nrof_faces))

        for face_position in bounding_boxes:

            try:
            
                face_position=face_position.astype(int)

                #print((int(face_position[0]), int( face_position[1])))
                #word_position.append((int(face_position[0]), int( face_position[1])))

                # Draw a rectangle around the faces


                crop=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
                # print(crop)
                crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC )

                data=crop.reshape(-1,160,160,3)

                prewhitened = prewhiten(data)

                emb_data = sess.run([embeddings],
                                    feed_dict={images_placeholder: np.array(prewhitened),
                                               phase_train_placeholder: False })[0]

                # 使用阈值作为判别

                # LABEL, PersonName, MinDistance = GetLabel(emb_data, train_x,train_y,Label2PersonDic)
                #
                #
                # if MinDistance < 0.95:
                #     cv2.putText(frame, '%s:%.3f' % (PersonName,MinDistance), (face_position[0], face_position[1] - 13),
                #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=2)
                # else:
                #     cv2.putText(frame, 'unknown:%.3f' % (MinDistance), (face_position[0], face_position[1] - 13),
                #                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=2)

                # use the knn classifier to discriminate

                kneighbors_distance, kneighbors_indexes = model.kneighbors(X=emb_data,n_neighbors=5, return_distance=True)




                predict = model.predict(emb_data)
                # print(predict)

                predict_prob = np.max(model.predict_proba(emb_data))

                non_zero_pro_num = np.count_nonzero(model.predict_proba(emb_data))




                # print(predict_prob)

                PersonName = Label2PersonDic[predict[0]]
                print('PersonName: ',PersonName)
                print(model.predict_proba(emb_data))
                print('prob var:',np.var(model.predict_proba(emb_data)))
                print('kneighbors_distance', kneighbors_distance)

                avr_distance = np.average(kneighbors_distance)
                print('avr_distance',avr_distance)

                print('non_zero_pro_num', non_zero_pro_num)

                print('\r\n\r\n\r\n')

                if ( predict_prob > 0.4 )and (avr_distance < 0.67)and(non_zero_pro_num<=2):

                    cv2.rectangle(frame, (face_position[0],
                                          face_position[1]),
                                  (face_position[2], face_position[3]),
                                  (0, 255, 0), 2)

                    cv2.putText(frame, '%s:%.3f' % (PersonName,predict_prob), (face_position[0], face_position[1] - 13),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=2)
                else:

                    cv2.rectangle(frame, (face_position[0],
                                          face_position[1]),
                                  (face_position[2], face_position[3]),
                                  (0, 0, 255), 2)

                    cv2.putText(frame, 'False' , (face_position[0], face_position[1] - 13),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), thickness=2, lineType=2)


            except :
                print('no face detected,and the error is :')
                traceback.print_exc()

        # cv2.putText(frame,'detected:{}'.format(find_results), (50,100),
        #         cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0),
        #         thickness = 2, lineType = 2)

    # print(faces)
    c += 1

    # Display the resulting frame
    cv2.imshow('Video', frame)
    # cv2.waitKey(200)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()



