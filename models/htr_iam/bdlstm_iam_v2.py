# Author: Tobi and Gundram


from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.rnn import bidirectional_rnn
from util.LoaderUtil import read_image_list, get_list_vals, clean_list
from util.CharacterMapper import get_cm_iam
from util.saver import PrefixSaver
from random import shuffle
import os
import time
import numpy as np
# import matplotlib.pyplot as plt
# import warnings

# Goes down to 10%
INPUT_PATH_TRAIN = './private/data/iam/lists/iam_train.lst'
INPUT_PATH_VAL = './private/data/iam/lists/iam_test.lst'
cm = get_cm_iam()
# Additional NaC Channel
nClasses = cm.size() + 1

nEpochs = 150
batchSize = 16
# learningRate = 0.001
# momentum = 0.9
# It is assumed that the TextLines are ALL saved with a consistent height of imgH
imgH = 32  # 64
# Depending on the size the image is skipped or zero padded
imgW = 2048  # 4096
image_depth = 1
nHiddenLSTM1 = 512
# Needs to be consistent with subsampling [X] in the model to correctly clean up the data
subsampling = 12

os.chdir("../..")

trainList = read_image_list(INPUT_PATH_TRAIN)
valList = read_image_list(INPUT_PATH_VAL)
print("Cleaning up train list:")
trainList = clean_list(trainList, imgW, cm, subsampling)
print("Cleaning up validation list:")
valList = clean_list(valList, imgW, cm, subsampling)

numT = 1024  # number of training samples per epoch
stepsPerEpochTrain = numT / batchSize
stepsPerEpochVal = len(valList) / batchSize


def inference(images, seqLen, keep_prob, phase_train):
    """

    :param images: tensor [batch][Y][X][Z] with dim(Z)=channels float32
    :param seqLen: tensor with length of batchsize containing the lenght of the images [batch] int32
    :param keep_prob: tensor with dim=0 dropout-rate float32
    :param phase_train: tensor with dim=0 boolean
    :return: output of network and length of output sequences after convolution [batch][1][x/subsample][channels+1] and [batch]
    """
    with tf.variable_scope('network'):
        with tf.variable_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([6, 5, image_depth, 32], stddev=5e-2), name='weights')
            # Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(images, kernel, [1, 4, 3, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[32]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1_bn = batch_norm(pre_activation, decay=0.999, is_training=phase_train, scope="BN1")
            conv1 = tf.nn.relu(conv1_bn, name=scope.name)
            norm1 = tf.nn.local_response_normalization(conv1, name='norm1')
            seqFloat = tf.to_float(seqLen)
            seqL2 = tf.ceil(seqFloat * 0.3333)
        with tf.variable_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=5e-2), name='weights')
            # # Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2_bn = batch_norm(pre_activation, decay=0.999, is_training=phase_train, scope="BN2")
            conv2 = tf.nn.relu(conv2_bn, name=scope.name)
            norm2 = tf.nn.local_response_normalization(conv2, name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='SAME', name='pool2')
            seqL3 = tf.ceil(seqL2 * 0.5)
        with tf.variable_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 3, 64, 128], stddev=5e-2), name='weights')
            # #Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[128]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3_bn = batch_norm(pre_activation, decay=0.999, is_training=phase_train, scope="BN3")
            conv3 = tf.nn.relu(conv3_bn, name=scope.name)
            norm3 = tf.nn.local_response_normalization(conv3, name='norm3')
            pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            seqL4 = tf.ceil(seqL3 * 0.5)
            # NO POOLING HERE -> CTC needs an appropriate length.
            seqLenAfterConv = tf.to_int32(seqL4)
        with tf.variable_scope('RNN_Prep') as scope:
            # (#batch Y X Z) --> (X #batch Y Z)
            rnnIn = tf.transpose(pool3, [2, 0, 1, 3])
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
            forwardH1 = rnn_cell.LSTMCell(nHiddenLSTM1,
                                          use_peepholes=True,
                                          state_is_tuple=True)
            droppedFW = rnn_cell.DropoutWrapper(forwardH1, output_keep_prob=keep_prob)
            backwardH1 = rnn_cell.LSTMCell(nHiddenLSTM1,
                                           use_peepholes=True,
                                           state_is_tuple=True)
            droppedBW = rnn_cell.DropoutWrapper(backwardH1, output_keep_prob=keep_prob)
            outputs, _, _ = bidirectional_rnn(droppedFW, droppedBW, rnnIn, dtype=tf.float32)
            fbH1rs = [tf.reshape(t, [batchSize, 2, nHiddenLSTM1]) for t in outputs]
            # outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]
            # eventually TODO instead of reduce_sum make matrix multiply
            outH1 = [tf.reduce_sum(t, reduction_indices=1) for t in fbH1rs]
        with tf.variable_scope('LOGIT') as scope:
            weightsClasses = tf.Variable(tf.truncated_normal([nHiddenLSTM1, nClasses],
                                                             stddev=np.sqrt(2.0 / nHiddenLSTM1)))
            biasesClasses = tf.Variable(tf.zeros([nClasses]))
            logitsFin = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

            logits3d = tf.pack(logitsFin)
    return logits3d, seqLenAfterConv


def loss(logits3d, tgt, seqLenAfterConv):
    loss = tf.reduce_sum(ctc.ctc_loss(logits3d, tgt, seqLenAfterConv))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.group(*update_ops)
        loss = control_flow_ops.with_dependencies([updates], loss)
    return loss


print('Defining graph')
graph = tf.Graph()
with graph.as_default():
    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=(batchSize, imgH, imgW, image_depth))
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))
    keep_prob = tf.placeholder(tf.float32)
    trainIN = tf.placeholder_with_default(tf.constant(False), [])
    logits3d, seqAfterConv = inference(inputX, seqLengths, keep_prob, trainIN)
    loss = loss(logits3d, targetY, seqAfterConv)
    saver = PrefixSaver('network', './private/models/iam_01/')
    # optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    # pred = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqAfterConv, merge_repeated=False)[0][0])
    pred = tf.to_int32(ctc.ctc_greedy_decoder(logits3d, seqAfterConv)[0][0])
    edist = tf.edit_distance(pred, targetY, normalize=False)
    tgtLens = tf.to_float(tf.size(targetY.values))
    err = tf.reduce_sum(edist) / tgtLens

with tf.Session(graph=graph) as session:
    # writer = tf.train.SummaryWriter('./log', session.graph)
    print('Initializing')
    tf.global_variables_initializer().run()
    for epoch in range(nEpochs):
        workList = trainList[:]
        shuffle(workList)
        workList = workList[0:numT]
        print('Epoch', epoch + 1, '...')
        lossT = 0
        errT = 0
        timeTS = time.time()
        tTL = 0
        for bStep in range(stepsPerEpochTrain):
            bList, workList = workList[:batchSize], workList[batchSize:]
            timeTemp = time.time()
            batchInputs, \
            batchSeqLengths, \
            batchTargetIdxs, \
            batchTargetVals, \
            batchTargetShape = get_list_vals(
                bList,
                cm,
                imgW,
                mvn=False)
            tTL += time.time() - timeTemp
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths, keep_prob: 0.5, trainIN: True}
            _, lossB, aErr = session.run([optimizer, loss, err], feed_dict=feedDict)
            lossT += lossB
            errT += aErr
        print('Train: CTC-loss ', lossT)
        cerT = errT / stepsPerEpochTrain
        print('Train: CER ', cerT)
        print('Train: time ', time.time() - timeTS)
        print('Time for loading train data: ', tTL)
        workList = valList[:]
        errV = 0
        lossV = 0
        timeVS = time.time()
        tVL = 0
        for bStep in range(stepsPerEpochVal):
            bList, workList = workList[:batchSize], workList[batchSize:]
            timeTemp = time.time()
            batchInputs, \
            batchSeqLengths, \
            batchTargetIdxs, \
            batchTargetVals, \
            batchTargetShape = get_list_vals(
                bList,
                cm,
                imgW,
                mvn=False
            )
            tVL += time.time() - timeTemp
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths, keep_prob: 1.0, trainIN: False}
            lossB, aErr = session.run([loss, err], feed_dict=feedDict)
            # lossB, aErr, sE, sL = session.run([loss, err, err_val, loss_val], feed_dict=feedDict)
            # writer.add_summary(sE, epoch*stepsPerEpocheVal + bStep)
            # writer.add_summary(sL, epoch * stepsPerEpocheVal + bStep)
            lossV += lossB
            errV += aErr
        print('Val: CTC-loss ', lossV)
        errVal = errV / stepsPerEpochVal
        print('Val: CER ', errVal)
        print('Val: time ', time.time() - timeVS)
        print('Time for loading validation data: ', tVL)
        # Write a checkpoint every 10 epochs
        if (epoch+1) % 10 == 0:
            saveTime =  time.time()
            print('Saving...')
            saver.save(session, global_step=epoch)
            print('Time for saving: ', time.time() - saveTime)