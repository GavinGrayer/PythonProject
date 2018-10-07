import tensorflow as tf
from tensorflow.contrib import rnn
import input_data
import numpy as np
import matplotlib.pyplot as plt

print("Packages imported")

mnist = input_data.read_data_sets("data/", one_hot=True)
trainimg, trainlabels, testimg, testlabels \
    = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
ntrain, ntest, dim, nclasses \
    = trainimg.shape[0], testimg.shape[0], trainimg.shape[1], trainlabels.shape[1]
print("MNIST loaded")

diminput = 28
dimhidden = 128
dimoutput = nclasses
#将28*28的图像切分为1*28的28个，分别输入不同的RNN单元
nsteps = 28
#初始化权重和偏置
weights = {
    'hidden':tf.Variable(tf.random_normal([diminput,dimhidden])),
    'out':tf.Variable(tf.random_normal([dimhidden,dimoutput]))
}
biases = {
    'hidden':tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}

def _RNN(_X,_W,_b,_nsteps,_name):
    #第一步：转换输入，输入_X是还有batchSize=5的5张28*28图片，需要将输入从
    #[batchSize,nsteps,diminput]==>[nsteps,batchSize,diminput]
    _X = tf.transpose(_X,[1,0,2])
#第二步：reshape _X为[nsteps*batchSize,diminput]
    _X = tf.reshape(_X,[-1,diminput])
#第三步：input layer -> hidden layer
    _H = tf.matmul(_X,_W['hidden'])+_b['hidden']
#第四步：将数据切分为‘nsteps’个切片，第i个切片为第i个batch data
    #tensoflow >0.12
    _Hsplit = tf.split(_H,_nsteps,0)
    #tensoflow <0.12  _Hsplit = tf.split(0,_nsteps,_H)
#第五步：计算LSTM final output(_LSTM_O) 和 state(_LSTM_S)
#_LSTM_O和_LSTM_S都有‘batchSize’个元素
#_LSTM_O用于预测输出
    with tf.variable_scope(_name) as scope:
        #scope.reuse_variables()
        # forget_bias = 1.0不忘记数据   
        ###tensorflow <1.0
        # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden,forget_bias = 1.0)
        #_LSTM_O,_SLTM_S = tf.nn.rnn(lstm_cell,_Hsplit,dtype=tf.float32)
        ###tensorflow 1.0
        lstm_cell = rnn.BasicLSTMCell(dimhidden)
        _LSTM_O,_LSTM_S = rnn.static_rnn(lstm_cell,_Hsplit,dtype=tf.float32)
#第六步：输出,需要最后一个RNN单元作为预测输出所以取_LSTM_O[-1]
        _O = tf.matmul(_LSTM_O[-1],_W['out'])+_b['out']
        return {'X':_X,'H':_H,'_Hsplit':_Hsplit,'LSTM_O':_LSTM_O,'LSTM_S':_LSTM_S,'O':_O}
print("Network Ready!")


if __name__ == '__main__':
    learning_rate = 0.001
    x = tf.placeholder("float", [None, nsteps, diminput])
    y = tf.placeholder("float", [None, dimoutput])
    myrnn = _RNN(x, weights, biases, nsteps, 'basic')
    pred = myrnn['O']
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)  # Adam
    accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
    init = tf.global_variables_initializer()
    print("Network Ready!")

    training_epochs = 5
    batch_size = 16
    display_step = 1
    sess = tf.Session()
    sess.run(init)
    print("Start optimization")
    for epoch in range(training_epochs):
        avg_cost = 0.
        # total_batch = int(mnist.train.num_examples/batch_size)
        total_batch = 100
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
            # print(batch_xs.shape)
            # print(batch_ys.shape)
            # batch_ys = batch_ys.reshape((batch_size, dimoutput))
            # Fit training using batch data
            feeds = {x: batch_xs, y: batch_ys}
            sess.run(optm, feed_dict=feeds)
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict=feeds) / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))

                feeds = {x: batch_xs, y: batch_ys}
                train_acc = sess.run(accr, feed_dict=feeds)
                print(" Training accuracy: %.3f" % (train_acc))
                testimgs = testimg.reshape((ntest, nsteps, diminput))
                feeds = {x: testimgs, y: testlabels}
                test_acc = sess.run(accr, feed_dict=feeds)
                print(" Test accuracy: %.3f" % (test_acc))
    print("Optimization Finished.")


