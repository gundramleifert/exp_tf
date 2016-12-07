'''

Author: Tobi and Gundram
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
from util.LoaderUtil import read_image_list, get_list_vals
from random import shuffle
from util.STR2CTC import get_charmap_lp
import os
import time
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH_TRAIN = './resources/lp_only_train.lst'
INPUT_PATH_VAL = './resources/lp_only_val.lst'
cm, nClasses = get_charmap_lp()
#Additional NaC Channel
nClasses += 1

nEpochs = 100
batchSize = 16
learningRate = 0.001
momentum = 0.9
#It is assumed that the TextLines are ALL saved with a consistent height of imgH
imgH = 20
#Depending on the size the image is cropped or zero padded
imgW = 100
channels = 1
nHiddenLSTM1 = 64
nHiddenLSTM2 = 64

os.chdir("..")
trainList = read_image_list(INPUT_PATH_TRAIN)
stepsPerEpocheTrain = len(trainList)/batchSize
valList = read_image_list(INPUT_PATH_VAL)
stepsPerEpocheVal = len(valList)/batchSize


def inference(images, seqLen):
  with tf.variable_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, channels, 16], stddev=5e-2), name = 'weights')
    ##Weight Decay?
    #weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
    #tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.1, shape=[16]), name='biases')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv1)
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                             padding='SAME', name='pool1')
    #norm1 = tf.nn.local_response_normalization(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
    seqFloat = tf.to_float(seqLen)
    seqL2 = tf.ceil(seqFloat * 0.5)
  with tf.variable_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 16, 64], stddev=5e-2), name='weights')
    ##Weight Decay?
    # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
    # tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    #_activation_summary(conv2)
    # norm2
    #norm2 = tf.nn.local_response_normalization(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1],strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    seqL3 = tf.ceil(seqL2 * 0.5)
  with tf.variable_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 128], stddev=5e-2), name='weights')
    ##Weight Decay?
    # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
    # tf.add_to_collection('losses', weight_decay)
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.1, shape=[128]), name='biases')
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    # _activation_summary(conv2)
    # norm2 = tf.nn.local_response_normalization(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
    #NO POOLING HERE -> CTC needs an appropriate length.
    seqLenAfterConv = tf.to_int32(seqL3)
  with tf.variable_scope('RNN_Prep') as scope:
    # (#batch Y X Z) --> (X #batch Y Z)
    rnnIn = tf.transpose(conv3, [2, 0, 1, 3])
    # (X #batch Y Z) --> (X #batch Y*Z)
    shape = rnnIn.get_shape()
    steps = shape[0]
    rnnIn = tf.reshape(rnnIn, tf.pack([shape[0], shape[1], -1]))
    # (X #batch Y*Z) --> (X*#batch Y*Z)
    shape = rnnIn.get_shape()
    rnnIn = tf.reshape(rnnIn, tf.pack([-1, shape[2]]))
    # (X*#batch Y*Z) --> list of X tensors of shape (#batch, Y*Z)
    rnnIn = tf.split(0, steps, rnnIn)
  with tf.variable_scope('BLSTM1') as scope:
    #Some kind of attention model -> switched it off for first tests
    #weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHiddenLSTM1],
    #                                                 stddev=np.sqrt(2.0 / (2 * nHiddenLSTM1))))
    #biasesOutH1 = tf.Variable(tf.zeros([nHiddenLSTM1]))
    forwardH1 = rnn_cell.LSTMCell(nHiddenLSTM1, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHiddenLSTM1, use_peepholes=True, state_is_tuple=True)
    outputs, _, _ = bidirectional_rnn(forwardH1, backwardH1, rnnIn, dtype=tf.float32)
    fbH1rs = [tf.reshape(t, [batchSize, 2, nHiddenLSTM1]) for t in outputs]
    #outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]
    outH1 = [tf.reduce_sum(t, reduction_indices=1) for t in fbH1rs]
  with tf.variable_scope('BLSTM2') as scope:
    # Some kind of attention model -> switched it off for first tests
    #weightsOutH2 = tf.Variable(tf.truncated_normal([2, nHiddenLSTM2],
    #                                                 stddev=np.sqrt(2.0 / (2 * nHiddenLSTM2))))
    #biasesOutH2 = tf.Variable(tf.zeros([nHiddenLSTM2]))
    forwardH2 = rnn_cell.LSTMCell(nHiddenLSTM2, use_peepholes=True, state_is_tuple=True)
    backwardH2 = rnn_cell.LSTMCell(nHiddenLSTM2, use_peepholes=True, state_is_tuple=True)
    outputs2, _, _ = bidirectional_rnn(forwardH2, backwardH2, outH1, dtype=tf.float32)

    fbH1rs2 = [tf.reshape(t, [batchSize, 2, nHiddenLSTM2]) for t in outputs2]
    #outH2 = [tf.reduce_sum(tf.mul(t, weightsOutH2), reduction_indices=1) + biasesOutH2 for t in fbH1rs2]
    outH2 = [tf.reduce_sum(t, reduction_indices=1) for t in fbH1rs2]
  with tf.variable_scope('LOGIT') as scope:
    weightsClasses = tf.Variable(tf.truncated_normal([nHiddenLSTM2, nClasses],
                                                       stddev=np.sqrt(2.0 / nHiddenLSTM1)))
    biasesClasses = tf.Variable(tf.zeros([nClasses]))
    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH2]
    logits3d = tf.pack(logits)
  return logits3d, seqLenAfterConv

def loss(logits3d, tgt, seqLenAfterConv):
  loss = tf.reduce_mean(ctc.ctc_loss(logits3d, tgt, seqLenAfterConv))
  return loss


print('Defining graph')
graph = tf.Graph()
with graph.as_default():
    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=(batchSize, imgH, imgW, channels))
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))
    logits3d, seqAfterConv = inference(inputX, seqLengths)
    loss = loss(logits3d, targetY, seqAfterConv)
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)
    pred = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqAfterConv)[0][0])
    err = tf.reduce_sum(tf.edit_distance(pred, targetY, normalize=False)) / tf.to_float(tf.size(targetY.values))
    # err_train = tf.scalar_summary('CER_TRAIN', err)
    # loss_train = tf.scalar_summary('LOSS_TRAIN', loss)
    # err_val = tf.scalar_summary('CER_VAL', err)
    # loss_val = tf.scalar_summary('LOSS_VAL', loss)


with tf.Session(graph=graph) as session:
    # writer = tf.train.SummaryWriter('./log', session.graph)
    print('Initializing')
    tf.initialize_all_variables().run()
    for epoch in range(nEpochs):
        workList = trainList[:]
        shuffle(workList)
        print('Epoch', epoch + 1, '...')
        lossE = 0
        errR = 0
        for bStep in range(stepsPerEpocheTrain):
            bList, workList = workList[:batchSize], workList[batchSize:]
            batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm, imgW, mvn=True)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            _, lossB, aErr = session.run([optimizer, loss, err], feed_dict=feedDict)
            # _, lossB, aErr, sET, sLT = session.run([optimizer, loss, err, err_train, loss_train], feed_dict=feedDict)
            lossE += lossB
            # writer.add_summary(sET, epoch * stepsPerEpocheTrain + bStep)
            # writer.add_summary(sLT, epoch * stepsPerEpocheTrain + bStep)
            #print(lossE)
            errR += aErr
        print('Train: CTC-loss ',lossE)
        cerT = errR/stepsPerEpocheTrain
        print('Train: CER ' , cerT)
        workList = valList[:]
        errV = 0
        for bStep in range(stepsPerEpocheVal):
            bList, workList = workList[:batchSize], workList[batchSize:]
            #print(bList)
            batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
                                                                                                             imgW,
                                                                                                             mvn=True)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
            lossB, aErr = session.run([loss, err], feed_dict=feedDict)
            # lossB, aErr, sE, sL = session.run([loss, err, err_val, loss_val], feed_dict=feedDict)
            # writer.add_summary(sE, epoch*stepsPerEpocheVal + bStep)
            # writer.add_summary(sL, epoch * stepsPerEpocheVal + bStep)
            lossE += lossB
            errR += aErr
        print('Val: CTC-loss ', lossE)
        errVal = errR / stepsPerEpocheVal
        print('Val: CER ', errVal)


