#!/usr/bin/env python
# -*- coding: utf-8 -*-
from PIL import Image
import struct
import matplotlib.pyplot as plt
def read_image(filename):
    plyt_img = []
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, images, rows, columns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')
    for i in range(800):
        image = Image.new('L', (columns, rows))
        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')
        print ('save ' + str(i) + 'image')
        image.save('./minist/pic/' + str(i) + '.png')
        plyt_img.append(image)
    return plyt_img

def read_label(filename, saveFilename):
    plyt_label = []
    f = open(filename, 'rb')
    index = 0
    buf = f.read()
    f.close()
    magic, labels = struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    #labelArr = [0] * labels
    labelArr = [0] * 800
    print(labels)
    for x in range(800):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
        save = open(saveFilename, 'w')
        save.write(','.join(map(lambda x: str(x), labelArr)))
        save.write('\n')
        save.close()
        plyt_label.append(labelArr[x])
    print ('save labels success')
    return plyt_label

if __name__ == '__main__':
    plyt_img = read_image('./input_data/train-images.idx3-ubyte')
    plyt_label = read_label('./input_data/train-labels.idx1-ubyte', './minist/label.txt')
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        title = u"标签对应为："+ str(plyt_label[i])
        plt.title(title, fontproperties='SimHei')
        plt.imshow(plyt_img[i], cmap='gray')
    plt.show()