import tensorflow as tf
import numpy as np
import random
def main():
    epochs=100000
    n=30
    dataX,dataY=create_data(n)
    trainX,trainY,testX,testY=divide_data(dataX,dataY,0.8)
    x=tf.placeholder(tf.float32,(None,n*n))
    y_=tf.placeholder(tf.float32,(None,2))
    w1=tf.Variable(tf.truncated_normal([n*n,2],stddev=0.1))
    b1=tf.Variable(tf.constant(0.1,shape=[2]))
    h1=tf.matmul(x,w1)+b1
    loss=tf.sqrt(tf.reduce_mean((h1-y_)**2))
    train_step=tf.train.MomentumOptimizer(0.05,0.01).minimize(loss)
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
def create_data(n):
    dataX = np.zeros((n*n,n*n))
    dataY = np.zeros((n*n,2))
    for i in xrange(n):
        for j in xrange(n):
            dataX[i*n+j,i*n+j]=1
            dataY[i*n+j,0]=i
            dataY[i*n+j,1]=j
    return dataX,dataY

if __name__=='__main__':
    main()