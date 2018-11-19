Real Time Face Recognition Fine-tuning By FaceNet
---

This is a real time face recognition demo(1:N in the face database ) fine-tuning by FaceNet using a Chinese Face Training Dataset spiding by **PIXEL COMPANY** and have a better performance than the pre-trained model provided by the FaceNet's author. 

Usage
---
1.you should register your face in the face database named `./FaceDataSet` by running `RegisterFace.py`


2.Run `Real time face detection and  recognition.py` to test the performance.

Details About the Training Set 
-----

This is a dataset spiding by PIXEL COMPANY which is not privided in public,including 7984 people(about 38850 samples).The example of one of the persons are showed as belowed:

![](https://i.imgur.com/VFLeHmq.png)

Details On the Fine-tuning paras
---

I use the source `train_tripletloss.py` provided by the FaceNet's author to fine-tuning the pre-trained model provided by him.

- **Input image size(aligned Face image)**

 `160*160*3`

- **embedding_size**

 `1*1*512`

- **images_per_person**

 `40`

- **batch_size**

 `90`

- **epoch_size**

 `500`

- **max_nrof_epochs**

 `500`

Performance of the fine-tunning model
---

### Test L2-Distance in the face images ###

- **Images**:

0: b1.png

1: b6.png

2: y1.png

3: y6.png

4: k1.png

5: s1.png

6: j1.jpg

7: j2.jpg

8: c1.png

9: c2.png

- **Distance Matrix**

Distance matrix

       0         1         2         3         4         5         6         7         8         9     
0    0.0000    0.1829    1.2259    1.2261    1.1326    1.1039    0.9905    0.9857    1.2346    1.4076  
1    0.1829    0.0000    1.2482    1.2475    1.1383    1.1056    1.0358    1.0373    1.2421    1.4111  
2    1.2259    1.2482    0.0000    0.1146    1.0772    1.1344    1.0892    1.0916    0.9325    1.0117  
3    1.2261    1.2475    0.1146    0.0000    1.0658    1.1275    1.1226    1.1223    0.9188    1.0004  
4    1.1326    1.1383    1.0772    1.0658    0.0000    0.9561    1.1950    1.1986    1.0749    1.2382  
5    1.1039    1.1056    1.1344    1.1275    0.9561    0.0000    1.0651    1.0700    1.1142    1.2993  
6    0.9905    1.0358    1.0892    1.1226    1.1950    1.0651    0.0000    0.1443    1.2670    1.2683  
7    0.9857    1.0373    1.0916    1.1223    1.1986    1.0700    0.1443    0.0000    1.2678    1.2682  
8    1.2346    1.2421    0.9325    0.9188    1.0749    1.1142    1.2670    1.2678    0.0000    0.9273  
9    1.4076    1.4111    1.0117    1.0004    1.2382    1.2993    1.2683    1.2682    0.9273    0.0000  




ori model

Distance matrix
        0         1         2         3         4         5         6         7         8         9     
0    0.0000    0.1378    1.0435    1.0528    0.9380    0.8837    1.1112    1.1011    1.0005    0.9732  
1    0.1378    0.0000    1.0400    1.0485    0.9404    0.8830    1.1114    1.1042    1.0048    0.9796  
2    1.0435    1.0400    0.0000    0.1265    1.0505    1.1035    1.0239    1.0140    0.9753    0.9692  
3    1.0528    1.0485    0.1265    0.0000    1.0518    1.1178    1.0379    1.0280    0.9686    0.9720  
4    0.9380    0.9404    1.0505    1.0518    0.0000    0.8731    1.1282    1.1159    0.9533    1.0348  
5    0.8837    0.8830    1.1035    1.1178    0.8731    0.0000    1.0611    1.0712    1.1097    1.0857  
6    1.1112    1.1114    1.0239    1.0379    1.1282    1.0611    0.0000    0.1318    1.0176    1.1431  
7    1.1011    1.1042    1.0140    1.0280    1.1159    1.0712    0.1318    0.0000    1.0147    1.1384  
8    1.0005    1.0048    0.9753    0.9686    0.9533    1.1097    1.0176    1.0147    0.0000    0.9370  
9    0.9732    0.9796    0.9692    0.9720    1.0348    1.0857    1.1431    1.1384    0.9370    0.0000  





### The loss in the training process ###
![](https://i.imgur.com/944OaXb.png)

Reference
----

1.MTCNN

2.FaceNet