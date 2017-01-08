from itertools import product

import time
from queue import Empty, Queue

from multiprocessing import Lock
from threading import Thread

from os.path import isfile

from cnn import CNNLearner, CNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def generateSeq(len, elements):
    rslt = [tuple([])]
    for i in range(len):
        rslt += list(product(elements, repeat=(i+1)))
    return rslt

def getTaskName(convs, fc):
    convs = ['x'.join(map(str, x)) for x in convs]
    return "cnn_" + '_'.join(convs) + "_fc_" + "_".join(map(str, fc))

def getFilename(convs, fc):
    return "cnn_models_results/" + getTaskName(convs, fc)

def learnMnistModel(convs=[(3, 3, 32), (3, 3, 64)], fc=[512]):
    start = time.time()
    filename = getFilename(convs, fc)
    cnn = CNN((784, ), (28, 28), 10, convs, fc)
    learner = CNNLearner(cnn, filename + '.txt', filename)
    init = tf.initialize_all_variables()
    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=sess_config)
    sess.run(init)
    acc = learner.learn(mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels, sess)
    dur = time.time() - start
    return (acc, dur)

def worker(coreName, tasksManager):
    try:
        while True:
            task = tasksManager.get()
            print (coreName, ":", getTaskName(task[0], task[1]))
            with tf.device(coreName):
               result = learnMnistModel(task[0], task[1])
            tasksManager.done(task, result)
    except(Empty):
        print("Queue is empty")

class TasksManager:
    def __init__(self, tasks):
        self.resultStrings = []
        self.lock = Lock()
        self.tasksQueue = Queue()
        self.fillTaskQueue(tasks)

    def done(self, task, result):
        self.lock.acquire()
        self.resultStrings += [getTaskName(task[0], task[1]) + "\t" + str(result) + "\n"]
        with open("cnn_results.txt", "w") as fd:
            fd.writelines(self.resultStrings)
        print("Queue size:", self.tasksQueue.qsize())
        self.lock.release()

    def fillTaskQueue(self, tasks):
        if (isfile("cnn_results.txt")):
            with open("cnn_results.txt", "r") as fd:
                self.resultStrings = fd.readlines()
        firstColumns = [x.split()[0] for x in self.resultStrings]
        for task in tasks:
            if not getTaskName(task[0], task[1]) in firstColumns:
                self.tasksQueue.put(task)

    def get(self):
        return self.tasksQueue.get_nowait()

def main():
    convSizes = [(3, 3), (5, 5), (7, 7)]
    convFilters = [16, 32, 64, 128]
    convs = [(x[0][0], x[0][1], x[1]) for x in product(convSizes, convFilters)]
    fc = [[512]]

    tasks = list(product(generateSeq(3, convs), fc))

    taskManager = TasksManager(tasks)

    th1 = Thread(target=worker, args=("/gpu:0", taskManager))
    th2 = Thread(target=worker, args=("/gpu:1", taskManager))
    th1.start()
    th2.start()
    th1.join()
    th2.join()

    #print(learnMnistModel(tasks[1]))

    # convs = [(5, 5, 64), (3, 3, 16)]
    # fcs = [ 512 ]
    # cnn = CNN((784, ), (28, 28), 10, convs, fcs)
    # filename = getFilename(convs, fcs)
    # learner = CNNLearner(cnn, filename + '.txt', filename)
    # init = tf.initialize_all_variables()
    # sess_config = tf.ConfigProto(allow_soft_placement=True)
    # sess = tf.Session(config=sess_config)
    # sess.run(init)
    # cnn.load(sess, getFilename(convs, fcs) + "_besttest")
    # print(learner.accurancy(mnist.test.images, mnist.test.labels, 100, sess))


if __name__ == '__main__':
    main()