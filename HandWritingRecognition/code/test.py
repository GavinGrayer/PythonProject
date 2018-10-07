from PIL import  Image
import random
import os
import numpy as np
import tensorflow as tf
import HandWritingRecognition.code.ReadImgAndLabel as ReadImgAndLabel
import matplotlib.pyplot as plt

number=['0','1','2','3','4','5','6','7','8','9']

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("label_dir", "./../minist/label.txt", "文件的目录")
tf.app.flags.DEFINE_string("img_dir",  "./../minist/pic/", "手写图片路径")

def convert2gray(img):
    '''
    图片转黑白   即3维转1维
    把彩色图像转为灰度图像（色彩对识别验证码没有什么用）
    :param img:
    :return:
    '''
    if len(img.shape)>2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        #r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        #gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text):
    '''
    验证码文本转向量
    :param text:
    :return:
    '''
    text_len = len(text)
    #print("text::",text,"text_len::",text_len,"max_label_len::",max_label_len)
    if text_len > max_label_len:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(max_label_len * char_set_len)

    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c) - 48
        if k > 9:
            k = ord(c) - 55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map')
        return k

    for i, c in enumerate(text):
        idx = i * char_set_len + char2pos(c)
        vector[idx] = 1
    return vector

#获得1组手写数据  1组128调数据    每一行的空间为一条验证码的存储空间
def get_next_batch(batch_size=128):
    batch_x=np.zeros([batch_size,image_height*image_width])
    batch_y=np.zeros([batch_size,max_label_len*char_set_len])

    def wrap_gen_captcha_text_and_image():
        while True:
            #print("while true.................")
            text, img = ReadImgAndLabel.get_mnist_text_image(FLAGS.label_dir, FLAGS.img_dir)
            if img.shape == (28,28):
                #print("into ...........")
                return text, img

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()

        #print("返回的值::text::",text," image::",image)
        image = convert2gray(image)

        #TODO
        #  将图片数组一维化 同时将文本也对应在两个二维组的同一行
        #[i, :] 第i行的所有数据
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y





def cnn_structure(w_alpha=0.01, b_alpha=0.1):
    with tf.variable_scope("conv1"):
        #对X进行形状的改变  由2维变4维  # 对x进行形状的改变[None, 28x28]  [None, 28, 28, 1]
        x = tf.reshape(X, shape=[-1, image_height, image_width, 1])

        #tf.contrib.layers.xavier_initializer()：一种经典的权值矩阵的初始化方式
        wc1=tf.get_variable(name='wc1',shape=[3,3,1,32],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

        #偏置
        bc1 = tf.Variable(b_alpha * tf.random_normal([32]))
        # [None, 28, 28, 1]-----> [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, wc1, strides=[1, 1, 1, 1], padding='SAME'), bc1))
        #池化 2x2 strides：2 [None,  28, 28, 32]---->[None, 14 ,14 , 32]  去冗余
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 防止过拟合
        conv1 = tf.nn.dropout(x_pool1, keep_prob)

    with tf.variable_scope("conv2"):
        wc2=tf.get_variable(name='wc2',shape=[3,3,32,64],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())

        bc2 = tf.Variable(b_alpha * tf.random_normal([64]))
        # [None, 14 ,14, 32]-----> [None, 14 ,14, 64]
        relu2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, wc2, strides=[1, 1, 1, 1], padding='SAME'), bc2))
        #池化 2x2 strides：2 [None, 14 ,14, 64]---->[None, 7 , 7 , 64]  去冗余
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        conv2 = tf.nn.dropout(pool2, keep_prob)
    # with tf.variable_scope("conv3"):
    #
    #     wc3=tf.get_variable(name='wc3',shape=[3,3,64,128],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
    #
    #     bc3 = tf.Variable(b_alpha * tf.random_normal([128]))
    #     #[None, 7 , 7, 64]-----> [None, 7 , 7, 128]
    #     relu3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, wc3, strides=[1, 1, 1, 1], padding='SAME'), bc3))
    #     #池化 2x2 strides：2 [None, 16, 40, 128]---->[None, 8, 20, 128]  去冗余
    #     pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #
    #     conv3 = tf.nn.dropout(pool3, keep_prob)

    with tf.variable_scope("fc"):
        wd1=tf.get_variable(name='wd1',shape=[7*7*64,1024],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
        #wd1 = tf.Variable(w_alpha * tf.random_normal([7*7*64,1024]))
        bd1 = tf.Variable(b_alpha * tf.random_normal([1024]))
        # 修改形状 [None, 7*7*64] --->[None, 7*7*64]
        dense = tf.reshape(conv2, [-1, wd1.get_shape().as_list()[0]])
        #[None, 7*7*64]*[7*7*64,1024] + [1024]  = [None , 1024 ]
        dense = tf.nn.relu(tf.add(tf.matmul(dense, wd1), bd1))
        dense = tf.nn.dropout(dense, keep_prob)

        with tf.variable_scope("outlayer"):
            #   [1024,1*10]
            wout=tf.get_variable('name',shape=[1024,max_label_len * char_set_len],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer())
            #0.1 * [1*10]
            bout = tf.Variable(b_alpha * tf.random_normal([max_label_len * char_set_len]))
            #   [None , 1024 ] * [1024,1*10] + 0.1*[1*10]   =   [None , 10]
            out = tf.add(tf.matmul(dense, wout), bout)
    return out



def train_cnn():
    out = cnn_structure()
    # 交叉熵计算loss 注意logits输入是在函数内部进行sigmod操作
    # sigmod_cross适用于每个类别相互独立但不互斥，如图中可以有字母和数字
    # softmax_cross适用于每个类别独立且排斥的情况，如数字和字母不可以同时出现
    loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out,labels=Y))

    # 最小化loss优化
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # 这里区分了大小写 实际上验证码一般不区分大小写
    # 预测值
    # [None , 40]    ----->>>    [None , 4 , 10 ]    存着结果值概率
    predict = tf.reshape(out, [-1, max_label_len, char_set_len])
    # axis = 2 是三维矩阵中每一个[]中的最大值下标
    max_idx_p = tf.argmax(predict, 2)
    # Y [None,4*10]  ---->>> [None,4,10]
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, max_label_len, char_set_len]), 2)

    correct_pred = tf.equal(max_idx_p, max_idx_l)
    # 将结果bool型转成0和1 的float型  并求出平均值，即预测结果和正确结果的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 定义一个初始化变量的op
        init = tf.global_variables_initializer()
        # 运行
        sess.run(init)
        step = 0
        while True:
            batch_x, batch_y = get_next_batch(100)
            _, cost_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            print("step::", step, " loss::", cost_)
            if step % 10 == 0:
                batch_x_test, batch_y_test = get_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("step::", step, " acc::", acc)
                if acc > 0.9:
                    saver.save(sess, "./../save/crack_capcha.model", global_step=step)
                    break
            step += 1


def crack_captcha(captcha_image):
    output = cnn_structure()

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "./../save/crack_capcha.model-80")

        predict = tf.argmax(tf.reshape(output, [-1, max_label_len, char_set_len]), 2)
        print(predict)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1.})
        print("text::" ,text_list)
        text = text_list[0].tolist()
        return text

if __name__ == '__main__':

    train = 1
    if train == 0:
        text, img = ReadImgAndLabel.get_mnist_text_image(FLAGS.label_dir,FLAGS.img_dir)
        image_height = 28
        image_width = 28
        max_label_len = len(text)
        print(max_label_len)
        char_set = number
        char_set_len = len(char_set)

        with tf.variable_scope("ready_data"):
            X = tf.placeholder(tf.float32, [None, image_height * image_width])
            Y = tf.placeholder(tf.float32, [None, max_label_len * char_set_len])
            keep_prob = tf.placeholder(tf.float32)
        train_cnn()

    if train == 1:
        text, image = ReadImgAndLabel.get_mnist_text_image(FLAGS.label_dir, FLAGS.img_dir)

        image_height = 28
        image_width = 28
        max_label_len = len(text)
        print(max_label_len)
        char_set = number
        char_set_len = len(char_set)
        max_captcha = len(text)

        X = tf.placeholder(tf.float32, [None, image_height * image_width])
        Y = tf.placeholder(tf.float32, [None, max_label_len * char_set_len])
        keep_prob = tf.placeholder(tf.float32)
        print("text::",text," image::",image)
        image_conv = convert2gray(image)
        image_conv = image_conv.flatten() / 255
        predict_text = crack_captcha(image_conv)

        print("正确: {}  预测: {}".format(text, predict_text))


        #作用新建绘画窗口,独立显示绘画的图片
        f = plt.figure()
        #将画布分割成1行1列，图像画在从左到右从上到下的第1块
        ax = f.add_subplot(111)
        #ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        title = u"标签对应为:" + str(text) + "  预测:" + str(predict_text)
        plt.title(title, fontproperties='SimHei')
        plt.imshow(image, cmap='gray')

        plt.show()
