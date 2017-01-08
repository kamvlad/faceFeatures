import threading

import tensorflow as tf
import numpy as np
import time

from math import ceil

from utils.batcher import Batcher

class CNN:
    def __init__(self, sampleSize, imageSize, outputs, conv = [(5, 5, 32), (5, 5, 64)], fc = [1024]):
        inputShape = [None] + list(sampleSize)
        self.outputSize = [outputs]
        self.inputs = tf.placeholder(tf.float32, shape=inputShape)
        self.reshapeInputs = tf.reshape(self.inputs, [-1] + list(imageSize) + [1])
        self.convLayers = []
        self.fullConnectedLayers = []

        self.toSaveVariables = []
        prevLayer = self.reshapeInputs
        prevConvSize = 1
        afterConvSize = imageSize
        for c in conv:
            convSize = (c[0], c[1], prevConvSize, c[2])
            weights = tf.Variable(tf.truncated_normal(convSize, stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[c[2]]))
            convolution = tf.nn.relu(tf.nn.conv2d(prevLayer, weights, strides=[1, 1, 1, 1], padding='SAME') + bias)
            pool = tf.nn.max_pool(convolution, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            self.toSaveVariables += [weights, bias]
            self.convLayers += [(weights, bias, convolution, pool)]
            prevConvSize = c[2]
            afterConvSize = (ceil(afterConvSize[0] / 2), ceil(afterConvSize[1] / 2))
            prevLayer = pool

        flatSize = int(afterConvSize[0] * afterConvSize[1]) * prevConvSize
        self.flat = tf.reshape(prevLayer, [-1, flatSize])
        prevLayerSize = flatSize
        prevLayer  = self.flat

        self.keep_prob = tf.placeholder(tf.float32)
        for f in fc:
            weights = tf.Variable(tf.truncated_normal([prevLayerSize, f], stddev=0.1))
            bias = tf.Variable(tf.constant(0.1, shape=[f]))
            fullConnectedLayer = tf.nn.relu(tf.matmul(prevLayer, weights) + bias)
            dropOut = tf.nn.dropout(fullConnectedLayer, self.keep_prob)

            self.toSaveVariables += [weights, bias]
            self.fullConnectedLayers += [(weights, bias, fullConnectedLayer, dropOut)]
            prevLayer = dropOut
            prevLayerSize = f

        weights = tf.Variable(tf.truncated_normal([prevLayerSize, outputs], stddev=0.1))
        bias = tf.Variable(tf.constant(0.1, shape=[outputs]))
        self.toSaveVariables += [weights, bias]
        self.outputLayer = [(weights, bias)]
        self.linearOutput = tf.matmul(prevLayer, weights) + bias
        self.output = tf.nn.softmax(self.linearOutput)
        self.saver = tf.train.Saver(self.toSaveVariables)

    def getFunction(self):
        return self.output

    def getLinearOutput(self):
        return self.linearOutput

    def save(self, session, filename):
        self.saver.save(sess=session, save_path=filename)

    def load(self, session, filename):
        self.saver.restore(sess=session, save_path=filename)

    def predict(self, data, sess):
        return self.output.eval(feed_dict={self.inputs: data, self.keep_prob: 1.0}, session=sess)

class CNNLearner:
    def __init__(self, net, reportFilename, bestScoreFilename):
        self.net = net
        self.bestScoreFilename = bestScoreFilename
        self.reportFilename = reportFilename
        self.targetOutput = tf.placeholder(tf.float32, shape = [None] + net.outputSize)
        self.lossFunc = tf.nn.softmax_cross_entropy_with_logits(net.getLinearOutput(), self.targetOutput)
        self.trainStep = tf.train.AdamOptimizer().minimize(self.lossFunc)

    def fit(self, trainData, trainLabels, session, keep_prob = 0.8, batchSize = 100):
        batcher = Batcher(trainData, trainLabels)
        batches = 0
        while batches < trainData.shape[0]:
            batchTrain, batchLabels = batcher.nextBatch(batchSize)
            session.run(self.trainStep, feed_dict = {self.net.inputs : batchTrain,
                                                     self.targetOutput : batchLabels,
                                                     self.net.keep_prob : keep_prob})
            batches += batchSize

    def learn(self, trainData, trainLabels, testData, testLabels, session, withoutMaxEpochsCount = 8,
              keep_prob = 0.8, batchSize = 100):
        maxTestAccurancy = 0.0
        maxTrainAccurancy = 0.0
        epoch = 0
        withoutMaxEpochs = 0
        report = []
        while withoutMaxEpochs < withoutMaxEpochsCount:
            self.fit(trainData, trainLabels, session, keep_prob)
            epoch += 1
            trainAccurancy = self.accurancy(trainData, trainLabels, batchSize, session)
            testAccurancy = self.accurancy(testData, testLabels, batchSize, session)

            if testAccurancy < maxTestAccurancy or trainAccurancy < maxTrainAccurancy:
                withoutMaxEpochs += 1

            if testAccurancy > maxTestAccurancy:
                self.net.save(session, self.bestScoreFilename + "_besttest")
                maxTestAccurancy = testAccurancy
                withoutMaxEpochs = 0
            if trainAccurancy > maxTrainAccurancy:
                self.net.save(session, self.bestScoreFilename + "_besttrain")
                maxTrainAccurancy = trainAccurancy
                withoutMaxEpochs = 0

            report += ["%d\t%f\t%f\t%d\t%f\t%f\n" % (epoch, testAccurancy, trainAccurancy, withoutMaxEpochs,
                                                     maxTestAccurancy, maxTrainAccurancy)]
            with open(self.reportFilename, 'w') as fd:
                fd.writelines(report)

        return (maxTestAccurancy, maxTrainAccurancy, epoch)

    def accurancy(self, testData, testLabels, batchSize, session):
        batches = int(testData.shape[0] / batchSize)
        idx = 0
        s = 0
        while batches > 0:
            predicted = self.net.predict(testData[idx:idx+batchSize], session)
            predictedLabels = np.argmax(predicted, axis=1)
            testLabelsArgMax = np.argmax(testLabels[idx:idx+batchSize], axis=1)
            s += np.sum(predictedLabels == testLabelsArgMax)
            idx += batchSize
            batches -= 1

        predicted = self.net.predict(testData[idx:], session)
        predictedLabels = np.argmax(predicted, axis=1)
        testLabelsArgMax = np.argmax(testLabels[idx:], axis=1)
        s += np.sum(predictedLabels == testLabelsArgMax)
        return s / testData.shape[0]