import tensorflow as tf
import numpy as np
import random

def main():
    epochs=1000

    m=19
    n=10
    dataX,dataY=create_data(m,n)
    trainX,trainY,testX,testY=divide_data(dataX,dataY,0.8)
    x=tf.placeholder(tf.float32,(None,m*n))
    x_mxn=tf.reshape(x,[-1,m,n,1])
    y_=tf.placeholder(tf.float32,(None,2))
    w1_column=tf.Variable(tf.truncated_normal([1,n,1,1],stddev=0.1))
    w1_row=tf.Variable(tf.truncated_normal([m,1,1,1],stddev=0.1))
    h1_column=tf.nn.conv2d(x_mxn,w1_column,[1,1,1,1],padding='VALID')
    h1_row=tf.nn.conv2d(x_mxn,w1_row,[1,1,1,1],padding='VALID')
    h1_row_sum=tf.reduce_sum(h1_row,2)
    h1_column_sum=tf.reduce_sum(h1_column,1)
    h1=tf.reshape(tf.concat(1,[h1_row_sum,h1_column_sum]),[-1,2])
    #b1=tf.Variable(tf.constant(0.1,shape=[2]))
    #h1=tf.matmul(x,w1)#+b1
    loss=tf.sqrt(tf.reduce_mean((h1-y_)**2))
    train_step=tf.train.AdamOptimizer(0.05).minimize(loss)
    sess=tf.InteractiveSession()
    saver=tf.train.Saver()
    #saver.restore(sess,'loc_sess')
    sess.run(tf.initialize_all_variables())
    for i in range(epochs):
        train_step.run(feed_dict={x:trainX,y_:trainY})
        if i%100==0:
            train_error=loss.eval(feed_dict={x:trainX,y_:trainY})
            test_error=loss.eval(feed_dict={x:testX,y_:testY})
            print 'train = ',train_error,', test = ',test_error

    print np.round(w1_column.eval()),np.round(w1_row.eval())
    saver.save(sess,'loc_sess')

def divide_data(dataX,dataY,ratio=0.8):
    n=dataX.shape[0]
    train_data_indices=random.sample(range(n),int(ratio*n))
    test_data_indices=list(set(range(n))-set(train_data_indices))
    trainX=dataX[train_data_indices]
    trainY=dataY[train_data_indices]
    testX=dataX[test_data_indices]
    testY=dataY[test_data_indices]
    return trainX,trainY,testX,testY
def create_data(m,n):
    dataX = np.zeros((m*n,m*n))
    dataY = np.zeros((m*n,2))
    for i in xrange(m):
        for j in xrange(n):
            dataX[i*n+j,i*n+j]=1
            dataY[i*n+j,0]=i
            dataY[i*n+j,1]=j
    return dataX,dataY

if __name__=='__main__':
    main()