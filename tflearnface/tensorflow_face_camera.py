#!/usr/bin/python
#coding=utf-8
''' face detect
https://github.com/seathiefwang/FaceRecognition-tensorflow
http://tumumu.cn/2017/05/02/deep-learning-face/
'''
# pylint: disable=invalid-name
import os
import random
import numpy as np
import cv2

def createdir(*args):
    ''' create dir'''
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)

IMGSIZE = 64


def getpaddingSize(shape):
    ''' get size to make image to be a square rect '''
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest]*4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()

def dealwithimage(img, h=64, w=64):
    ''' dealwithimage '''
    #img = cv2.imread(imgpath)
    top, bottom, left, right = getpaddingSize(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img


def relight(imgsrc, alpha=1, bias=0):
    '''relight
        改变图片的亮度与对比度
    '''
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc

def getfacefromcamera(outdir):
    #创建路径
    createdir(outdir)

    camera = cv2.VideoCapture(0)
    #Haar特征分类器就是一个XML文件，该文件中会描述人体各个部位的Haar特征值。包括人脸、眼睛、嘴唇等等
    haar = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    n = 1
    while 1:
        if (n <= 200):
            print('It`s processing %s image.' % n)
            # 读帧
            success, img = camera.read()
            #灰度转换的作用就是：转换成灰度的图片的计算强度得以降低
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #opencv2中人脸检测使用的是 detectMultiScale函数。它可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小（用矩形表示），函数由分类器对象调用
            '''
            参数1：image--待检测图片，一般为灰度图像加快检测速度；
            参数2：objects--被检测物体的矩形框向量组；
            参数3：scaleFactor--表示在前后两次相继的扫描中，搜索窗口的比例系数。默认为1.1即每次搜索窗口依次扩大10%;
            参数4：minNeighbors--表示构成检测目标的相邻矩形的最小个数(默认为3个)。        
            如果组成检测目标的小矩形的个数和小于 min_neighbors - 1 都会被排除。        
            如果min_neighbors 为 0, 则函数不做任何操作就返回所有的被检候选矩形框，        
            这种设定值一般用在用户自定义对检测结果的组合程序上；

            '''
            faces = haar.detectMultiScale(gray_img, 1.3, 5)
            for f_x, f_y, f_w, f_h in faces:
                face = img[f_y:f_y+f_h, f_x:f_x+f_w]
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                #could deal with face to train
                face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
                cv2.imwrite(os.path.join(outdir, str(n)+'.jpg'), face)

                cv2.putText(img, 'haha', (f_x, f_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)  #显示名字
                img = cv2.rectangle(img, (f_x, f_y), (f_x + f_w, f_y + f_h), (255, 0, 0), 2)
                n+=1
            cv2.imshow('img', img)
            key = cv2.waitKey(30) & 0xff
            if key == 27:
                break
        else:
            break
    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    name = input('please input yourename: ')
    getfacefromcamera(os.path.join('./image/trainfaces', name))

