# coding: utf-8

import cv2
import os
from os.path import join as pjoin
# import tensorflow as tf
# import numpy as np
import net
import traceback

video_capture = cv2.VideoCapture(0)
frame_interval = 5

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

FACE_NUM_PER_PERSON = 7

#路径
dataSetRootDir = './FaceDataSet/'

# restore mtcnn model

# print('Creating mtcnn networks and loading parameters')
# gpu_memory_fraction=0.7
# with tf.Graph().as_default():
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction,allow_growth=True)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#     with sess.as_default():
#         pnet, rnet, onet = net.create_mtcnn(sess, './model_check_point/')
# print('MTCNN networks created')


personname = input('PersonName: ')
print('Please press key \'s\' to start register ur face: ')
picNum = 0
RegisteSwitch = 0
while True:
    # Capture frame-by-frame

    ret, frame = video_capture.read()

    timeF = frame_interval

    frame2show = []
    frame2show = frame

    cv2.putText(frame2show, 'Press key \'s\' to register ur face 5 times', (20, 20),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=2)
    cv2.putText(frame2show, 'faceNum:%d'%(picNum), (20,40),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=2)
    cv2.imshow('Video', frame2show)

    if (cv2.waitKey(1) & 0xFF == ord('s')):
        RegisteSwitch = 1

    if RegisteSwitch == 1:
        persondir = pjoin(dataSetRootDir, personname)
        if not os.path.isdir(persondir):  # Create the person id directory if it doesn't exist
            os.makedirs(persondir)

        cv2.imwrite(persondir + '/' + personname + '_' + str(picNum) + '.jpg', frame)
        picNum += 1

    if (cv2.waitKey(1) & 0xFF == ord('q')) or (picNum == FACE_NUM_PER_PERSON):
        break

# When everything is done, release the capture

video_capture.release()
cv2.destroyAllWindows()

# Register And Crop Face by MTCNN SALUTION

# video_capture = cv2.VideoCapture(0)
# frame_interval = 5
#
# #face detection parameters
# minsize = 20 # minimum size of face
# threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
# factor = 0.709 # scale factor
#
# FACE_NUM_PER_PERSON = 5
#
# #路径
# dataSetRootDir = './FaceDataSet/'
#
# # restore mtcnn model
#
# print('Creating mtcnn networks and loading parameters')
# gpu_memory_fraction=0.7
# with tf.Graph().as_default():
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction,allow_growth=True)
#     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
#     with sess.as_default():
#         pnet, rnet, onet = net.create_mtcnn(sess, './model_check_point/')
# print('MTCNN networks created')
#
#
# personname = input('PersonName: ')
# print('Please press key \'s\' to start register ur face: ')
# picNum = 0
# RegisteSwitch = 0
# while True:
#     # Capture frame-by-frame
#
#     ret, frame = video_capture.read()
#
#     timeF = frame_interval
#
#     frame2show = []
#     frame2show = frame
#
#     cv2.putText(frame2show, 'Press key \'s\' to register ur face 5 times', (20, 20),
#                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=2)
#     cv2.putText(frame2show, 'faceNum:%d'%(picNum), (20,40),
#                 cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), thickness=2, lineType=2)
#     cv2.imshow('Video', frame2show)
#
#     if (cv2.waitKey(1) & 0xFF == ord('s')):
#         RegisteSwitch = 1
#
#     if RegisteSwitch == 1:
#         persondir = pjoin(dataSetRootDir, personname)
#         if not os.path.isdir(persondir):  # Create the log directory if it doesn't exist
#             os.makedirs(persondir)
#         #detect and crop the face in the frame
#         bounding_boxes, _ = net.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
#         nrof_faces = bounding_boxes.shape[0]  # number of faces
#         print('nrof_faces  ',nrof_faces)
#         print('bounding_boxes',bounding_boxes.shape)
#         print(bounding_boxes)
#
#         for face_position in bounding_boxes:
#             try:
#                 # 只有当人脸框的置信概率大于0.8才把这个人脸crop下来
#                 if face_position[4] >= 0.8:
#                     # 将位置坐标转成Int
#                     face_position = face_position.astype(int)
#                     crop = frame[face_position[1]:face_position[3],
#                            face_position[0]:face_position[2], ]
#                     # cv2.imshow('crop', crop)
#                     crop_and_resize = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC)
#                     # cv2.imshow('crop_and_resize', crop_and_resize)
#                     cv2.imwrite(persondir + '/' + personname + '_' + str(picNum) + '.jpg', crop_and_resize)
#                     picNum += 1
#
#             except :
#                 traceback.print_exc()
#
#
#
#
#     if (cv2.waitKey(1) & 0xFF == ord('q')) or (picNum == FACE_NUM_PER_PERSON):
#         break
#
# # When everything is done, release the capture
#
# video_capture.release()
# cv2.destroyAllWindows()