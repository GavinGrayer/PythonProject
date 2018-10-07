from PIL import  Image
import random
import os
import numpy as np
import matplotlib.pyplot as plt

number=['0','1','2','3','4','5','6','7','8','9']


def get_mnist_text(filepath,index):
    '''
    返回字符串   对应图片的数字
    :param filepath:
    :return:
    '''
    labels_list = []
    #print("index::",index)
    with open(filepath) as file:
        labels = file.read()
        labels_list = labels.strip(',').split(',')
        #print(labels_list)


    return labels_list[index]

def past_get_mnist_text_image(text_dir,img_dir):

    text_list = get_mnist_text(text_dir)

    img_list = []
    for img_name in os.listdir(img_dir):

        img_path = img_dir + img_name  # 每一个图片的地址
        #print(img_path)
        img = Image.open(img_path)
        #img = img.resize((28,28))  # 将图片保存成224×224大小
        # 转成np数组
        np.set_printoptions(threshold=np.NaN)  # 全部输出
        img = np.array(img)
        img_list.append(img)
        #print(img.shape)

    return text_list,img_list

def get_mnist_text_image(text_dir,img_dir):

    index = random.randint(0,799)
    text_list = get_mnist_text(text_dir,index)
    img_list = []

    img_path = img_dir + str(index) + ".png"  # 每一个图片的地址
    #print(img_path)
    img = Image.open(img_path)
    #img = img.resize((28,28))  # 将图片保存成224×224大小
    # 转成np数组
    np.set_printoptions(threshold=np.NaN)  # 全部输出
    img = np.array(img)
    img_list.append(img)
    #print(img.shape)

    return text_list,img




if __name__ == '__main__':
    text_list, img_list = get_mnist_text_image("./../minist/label.txt","./../minist/pic/")
    print(text_list,img_list)
    plt.subplot(1, 1, 1)
    title = u"标签对应为：" + str(text_list)
    plt.imshow(img_list, cmap='gray')
    plt.show()
    # for i in range(20):
    #     print("==========%d",(i))
    #     print(type(text_list[i]))
    #     print(type(img_list[i]))