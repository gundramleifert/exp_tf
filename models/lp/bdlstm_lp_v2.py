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
from util.STR2CTC import get_charmap_lp, get_charmap_lp_inv
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Goes done to 10%
INPUT_PATH_TRAIN = './private/lists/lp_only_train.lst'
INPUT_PATH_VAL = './private/lists/lp_only_val.lst'
cm, nClasses = get_charmap_lp()
# Additional NaC Channel
nClasses += 1

nEpochs = 100
batchSize = 1
learningRate = 0.001
momentum = 0.9
# It is assumed that the TextLines are ALL saved with a consistent height of imgH
imgH = 48
# Depending on the size the image is cropped or zero padded
imgW = 256
channels = 1
nHiddenLSTM1 = 256
nHiddenLSTM2 = 256
nHiddenInner = 256
dropout = 0.5

os.chdir("../..")
trainList = read_image_list(INPUT_PATH_TRAIN)
stepsPerEpocheTrain = len(trainList) / batchSize
valList = read_image_list(INPUT_PATH_VAL)
stepsPerEpocheVal = len(valList) / batchSize


def inference(images, seqLen):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([4, 4, channels, 42], stddev=5e-2), name='weights')
        ##Weight Decay?
        # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(images, kernel, [1, 2, 2, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[42]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # _activation_summary(conv1)
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1')
        # norm1 = tf.nn.local_response_normalization(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        seqFloat = tf.to_float(seqLen)
        seqL2 = tf.ceil(seqFloat * 0.25)
    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 42, 64], stddev=5e-2), name='weights')
        ##Weight Decay?
        # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        # _activation_summary(conv2)
        # norm2
        # norm2 = tf.nn.local_response_normalization(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='SAME', name='pool2')
        seqL3 = tf.ceil(seqL2 * 0.5)
    with tf.variable_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 3, 64, 256], stddev=5e-2), name='weights')
        ##Weight Decay?
        # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[256]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='SAME', name='pool2')
        # NO POOLING HERE -> CTC needs an appropriate length.
        seqLenAfterConv = tf.to_int32(seqL3)
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
        forwardH1 = rnn_cell.LSTMCell(nHiddenLSTM1, use_peepholes=True, state_is_tuple=True)
        backwardH1 = rnn_cell.LSTMCell(nHiddenLSTM1, use_peepholes=True, state_is_tuple=True)
        outputs, _, _ = bidirectional_rnn(forwardH1, backwardH1, rnnIn, dtype=tf.float32)
        fbH1rs = [tf.reshape(t, [batchSize, 2, nHiddenLSTM1]) for t in outputs]
        # outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]
        outH1 = [tf.reduce_sum(t, reduction_indices=1) for t in fbH1rs]
    with tf.variable_scope('BLSTM2') as scope:
        forwardH2 = rnn_cell.LSTMCell(nHiddenLSTM2, use_peepholes=True, state_is_tuple=True)
        backwardH2 = rnn_cell.LSTMCell(nHiddenLSTM2, use_peepholes=True, state_is_tuple=True)
        outputs2, _, _ = bidirectional_rnn(forwardH2, backwardH2, rnnIn, dtype=tf.float32)
        fbH1rs2 = [tf.reshape(t, [batchSize, 2, nHiddenLSTM2]) for t in outputs2]
        # outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]
        outH2 = [tf.reduce_sum(t, reduction_indices=1) for t in fbH1rs2]
    with tf.variable_scope('LOGIT') as scope:
        weightsHid = tf.Variable(tf.truncated_normal([nHiddenLSTM1, nHiddenInner],
                                                    stddev=np.sqrt(2.0 / nHiddenLSTM1)))
        biasesHid = tf.Variable(tf.zeros([nHiddenInner]))
        logits = [tf.matmul(t, weightsHid) + biasesHid for t in outH2]
        acti = [tf.nn.relu(t) for t in logits]
        dropped = [tf.nn.dropout(t, dropout) for t in acti]

        weightsClasses = tf.Variable(tf.truncated_normal([nHiddenLSTM2, nClasses],
                                                         stddev=np.sqrt(2.0 / nHiddenLSTM2)))
        biasesClasses = tf.Variable(tf.zeros([nClasses]))
        logitsFin = [tf.matmul(t, weightsClasses) + biasesClasses for t in dropped]

        logits3d = tf.pack(logitsFin)
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
    # pred = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqAfterConv, merge_repeated=False)[0][0])
    pred = tf.to_int32(ctc.ctc_greedy_decoder(logits3d, seqAfterConv)[0][0])
    edist = tf.edit_distance(pred, targetY, normalize=False)
    tgtLens = tf.to_float(tf.size(targetY.values))
    err = tf.reduce_sum(edist) / tgtLens
    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    # writer = tf.train.SummaryWriter('./log', session.graph)
    print('Initializing')
    tf.global_variables_initializer().run()

    ckpt = tf.train.get_checkpoint_state("./private/models/lp2/")
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
    print(ckpt)
    workList = valList[:]
    errV = 0
    lossV = 0
    timeVS = time.time()
    cmInv = get_charmap_lp_inv()
    for bStep in range(stepsPerEpocheVal):
        bList, workList = workList[:batchSize], workList[batchSize:]
        batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
                                                                                                           imgW,
                                                                                                         mvn=True)
        feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                    targetShape: batchTargetShape, seqLengths: batchSeqLengths}
        lossB, aErr, p = session.run([loss, err, pred], feed_dict=feedDict)
        print(aErr)
        res = []
        for idx in p.values:
            res.append(cmInv[idx])
        print(res)
        # print(p)
        plt.imshow(batchInputs[0,:,:,0], cmap=plt.get_cmap("gray"))
        plt.show()

        lossV += lossB
        errV += aErr
    print('Val: CTC-loss ', lossV)
    errVal = errV / stepsPerEpocheVal
    print('Val: CER ', errVal)
    print('Val time ', time.time() - timeVS)
    # for epoch in range(nEpochs):
    #     workList = trainList[:]
    #     shuffle(workList)
    #     print('Epoch', epoch + 1, '...')
    #     lossT = 0
    #     errT = 0
    #     timeTS = time.time()
    #     for bStep in range(stepsPerEpocheTrain):
    #         bList, workList = workList[:batchSize], workList[batchSize:]
    #         batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
    #                                                                                                          imgW,
    #                                                                                                          mvn=True)
    #         feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
    #                     targetShape: batchTargetShape, seqLengths: batchSeqLengths}
    #         _, lossB, aErr = session.run([optimizer, loss, err], feed_dict=feedDict)
    #         # _, lossB, aErr, sET, sLT = session.run([optimizer, loss, err, err_train, loss_train], feed_dict=feedDict)
    #         lossT += lossB
    #         # writer.add_summary(sET, epoch * stepsPerEpocheTrain + bStep)
    #         # writer.add_summary(sLT, epoch * stepsPerEpocheTrain + bStep)
    #         errT += aErr
    #     print('Train: CTC-loss ', lossT)
    #     cerT = errT / stepsPerEpocheTrain
    #     print('Train: CER ', cerT)
    #     print('Train time ', time.time() - timeTS)
    #     workList = valList[:]
    #     errV = 0
    #     lossV = 0
    #     timeVS = time.time()
    #     for bStep in range(stepsPerEpocheVal):
    #         bList, workList = workList[:batchSize], workList[batchSize:]
    #         batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
    #                                                                                                          imgW,
    #                                                                                                          mvn=True)
    #         feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
    #                     targetShape: batchTargetShape, seqLengths: batchSeqLengths}
    #         lossB, aErr = session.run([loss, err], feed_dict=feedDict)
    #         # lossB, aErr, sE, sL = session.run([loss, err, err_val, loss_val], feed_dict=feedDict)
    #         # writer.add_summary(sE, epoch*stepsPerEpocheVal + bStep)
    #         # writer.add_summary(sL, epoch * stepsPerEpocheVal + bStep)
    #         lossV += lossB
    #         errV += aErr
    #     print('Val: CTC-loss ', lossV)
    #     errVal = errV / stepsPerEpocheVal
    #     print('Val: CER ', errVal)
    #     print('Val time ', time.time() - timeVS)
    #     # Write a checkpoint.
    #     checkpoint_file = os.path.join('./private/models/lp2/', 'checkpoint')
    #     saver.save(session, checkpoint_file, global_step=epoch)
