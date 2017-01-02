'''

Author: Tobi and Gundram
'''

from __future__ import print_function

from itertools import chain

import tensorflow as tf
from util.spatial_transformer import transformer
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.rnn import bidirectional_rnn
from util.LoaderUtil import read_image_list, get_list_vals
from util.saver import PrefixSaver
from util.saver import get_op
from util.CharacterMapper import get_cm_lp
from util.variables import get_uninitialized_variables
from random import shuffle

import os
import time
import numpy as np
import matplotlib.pyplot as plt

nEpochs = 1000
batchSize = 16
# Goes done to 10%
INPUT_PATH_TRAIN = './private/lists/lp0.lst'
INPUT_PATH_TRAIN1 = './private/lists/lp1.lst'
INPUT_PATH_TRAIN2 = './private/lists/lp2.lst'
INPUT_PATH_TRAIN3 = './private/lists/lp3.lst'
INPUT_PATH_TRAIN4 = './private/lists/lp4.lst'
INPUT_PATH_TRAIN5 = './private/lists/lp_enlarge_train.lst'
INPUT_PATH_VAL = './private/lists/lp_enlarge_val.lst'
cm = get_cm_lp()
# Additional NaC Channel
nClasses = cm.size() + 1
os.chdir("../..")
trainList = read_image_list(INPUT_PATH_TRAIN)
trainList1 = read_image_list(INPUT_PATH_TRAIN1)
trainList2 = read_image_list(INPUT_PATH_TRAIN2)
trainList3 = read_image_list(INPUT_PATH_TRAIN3)
trainList4 = read_image_list(INPUT_PATH_TRAIN4)
trainList5 = read_image_list(INPUT_PATH_TRAIN5)
numT = 16000
stepsPerEpocheTrain = numT / batchSize
valList = read_image_list(INPUT_PATH_VAL)
stepsPerEpocheVal = len(valList) / batchSize

channels = 1
learningRate = 0.001
momentum = 0.9
#####This is the image size, for the READ part of the net.... so the output of the last STN
# It is assumed that the TextLines are ALL saved with a consistent height of imgH
imgH = 48
# Depending on the size the image is cropped or zero padded
imgW = 256
nHiddenLSTM1 = 256

imgInW = 512
imgInH = 512


def inference(images, seqLen, keep_prob, phase_train):
    with tf.variable_scope('findPart') as scope:
        imagesRes = tf.image.resize_bilinear(images, (128,128))
        with tf.variable_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, channels, 32], stddev=5e-2), name='weights')
            ##Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(imagesRes, kernel, [1, 2, 2, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[32]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.tanh(pre_activation, name=scope.name)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        with tf.variable_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 128], stddev=5e-2), name='weights')
            ##Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(pool1, kernel, [1, 2, 2, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[128]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.tanh(pre_activation, name=scope.name)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        with tf.variable_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 128, 512], stddev=5e-2), name='weights')
            ##Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(pool2, kernel, [1, 2, 2, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[512]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.tanh(pre_activation, name=scope.name)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            h_conv3_flat = tf.reshape(pool3, [batchSize, 2048])
        with tf.variable_scope('ff1') as scope:
            # h_conv2_flat = tf.reshape(imagesRes, [batchSize, 16384])
            # W_fc_loc1 = tf.Variable(tf.zeros([128,20]), name='weights')
            # b_fc_loc1 = tf.Variable(tf.zeros([20]), name='bias')
            # W_fc_loc2 = tf.Variable(tf.zeros([20,6]), name='weights')
            W_fc_loc1 = tf.Variable(tf.truncated_normal([2048, 20], stddev=5e-4), name='weights')
            b_fc_loc1 = tf.Variable(tf.truncated_normal([20], stddev=5e-4), name='bias')
            W_fc_loc2 = tf.Variable(tf.truncated_normal([20, 4], stddev=5e-4), name='weights')
            # Use identity transformation as starting point
            # initial = np.array([[0.5, 0, 128], [0, 0.1, 232]])

            # s_x = tf.Variable(1.0, name='s_x',dtype='float32')
            s_x = tf.Variable(0.5, name='s_x',dtype='float32')
            # s_y = tf.Variable(1.0, name='s_y',dtype='float32')
            s_y = tf.Variable(0.093, name='s_y',dtype='float32')
            t_x = tf.Variable(0.0, name='t_x',dtype='float32')
            t_y = tf.Variable(0.0, name='t_y',dtype='float32')
            # r_1 = tf.constant(0.0, name="r_1")
            # r_2 = tf.constant(0.0, name="r_2")
            # b_fc_loc2 = [s_x, r_1, t_x, r_2, s_y, t_y]
            b_fc_loc2 = [s_x, t_x, s_y, t_y]


            # b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')
            # %% Define the two layer localisation network
            h_fc_loc1 = tf.nn.tanh(tf.matmul(h_conv3_flat, W_fc_loc1) + b_fc_loc1)
            # %% We can add dropout for regularizing and to reduce overfitting like so:
            h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, keep_prob)
            # %% Second layer
            h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)
            # print(h_fc_loc2[:,0].get_shape())
            # print(tf.constant(0,dtype='float32', shape=[batchSize]).get_shape())
            aff = [h_fc_loc2[:,0], tf.constant(0,dtype='float32', shape=[batchSize]), h_fc_loc2[:,1], tf.constant(0,dtype='float32', shape=[batchSize]), h_fc_loc2[:,2], h_fc_loc2[:,3]]
            # print(aff)
            finAff = tf.pack(aff)
            finAff = tf.transpose(finAff, [1, 0])
            # print(finAff.get_shape())

            # %% We'll create a spatial transformer module to identify discriminative
            # %% patches
            out_size = (imgH, imgW)
            stn_out = transformer(images, finAff, out_size)
            stn_out = tf.reshape(stn_out, [batchSize, imgH, imgW, 1])
            mean, var = tf.nn.moments(stn_out, axes=[1,2], keep_dims=True)
            # print(mean.get_shape())
            stn_out = tf.nn.batch_normalization(stn_out, mean=mean, variance=var, offset=None, scale=None, variance_epsilon=1e-6)
    with tf.variable_scope('readPart') as scope:
        with tf.variable_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([6, 5, channels, 32], stddev=5e-2), name='weights')
            ##Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(stn_out, kernel, [1, 4, 3, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[32]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1_bn = batch_norm(pre_activation, decay=0.999, is_training=phase_train, scope="BN1")
            conv1 = tf.nn.relu(conv1_bn, name=scope.name)
            norm1 = tf.nn.local_response_normalization(conv1, name='norm1')
            # _activation_summary(conv1)
            # norm1 = tf.nn.local_response_normalization(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
            seqFloat = tf.to_float(seqLen)
            seqL2 = tf.ceil(seqFloat * 0.33)
        with tf.variable_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=5e-2), name='weights')
            ##Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2_bn = batch_norm(pre_activation, decay=0.999, is_training=phase_train, scope="BN2")
            conv2 = tf.nn.relu(conv2_bn, name=scope.name)
            norm2 = tf.nn.local_response_normalization(conv2, name='norm2')
            # _activation_summary(conv2)
            # norm2
            # norm2 = tf.nn.local_response_normalization(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='SAME', name='pool2')
            seqL3 = tf.ceil(seqL2 * 0.5)
        with tf.variable_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 3, 64, 128], stddev=5e-2), name='weights')
            ##Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[128]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv3_bn = batch_norm(pre_activation, decay=0.999, is_training=phase_train, scope="BN3")
            conv3 = tf.nn.relu(conv3_bn, name=scope.name)
            norm3 = tf.nn.local_response_normalization(conv3, name='norm3')
            pool3 = tf.nn.max_pool(norm3, ksize=[1, 3, 1, 1], strides=[1, 3, 1, 1], padding='SAME', name='pool2')
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
            droppedFW = rnn_cell.DropoutWrapper(forwardH1, output_keep_prob=keep_prob)
            backwardH1 = rnn_cell.LSTMCell(nHiddenLSTM1, use_peepholes=True, state_is_tuple=True)
            droppedBW = rnn_cell.DropoutWrapper(backwardH1, output_keep_prob=keep_prob)
            outputs, _, _ = bidirectional_rnn(droppedFW, droppedBW, rnnIn, dtype=tf.float32)
            fbH1rs = [tf.reshape(t, [batchSize, 2, nHiddenLSTM1]) for t in outputs]
            # outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]
            outH1 = [tf.reduce_sum(t, reduction_indices=1) for t in fbH1rs]
        with tf.variable_scope('LOGIT') as scope:

            weightsClasses = tf.Variable(tf.truncated_normal([nHiddenLSTM1, nClasses],
                                                             stddev=np.sqrt(2.0 / nHiddenLSTM1)))
            biasesClasses = tf.Variable(tf.zeros([nClasses]))
            logitsFin = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

            logits3d = tf.pack(logitsFin)
    return logits3d, seqLenAfterConv, stn_out, imagesRes, finAff


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
    inputX = tf.placeholder(tf.float32, shape=(batchSize, imgInH, imgInW, channels))
    # inputX = tf.placeholder(tf.float32, shape=(batchSize, imgH, imgW, channels))
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    # seqLengths = tf.placeholder(tf.int32, shape=(batchSize))
    seqLengths = imgW*tf.ones(shape=(batchSize))
    keep_prob = tf.placeholder(tf.float32)
    trainIN = tf.placeholder_with_default(tf.constant(False), [])
    logits3d, seqAfterConv, stn_o, img_sub, f_aff = inference(inputX, seqLengths, keep_prob, trainIN)
    loss = loss(logits3d, targetY, seqAfterConv)
    # saver1 = PrefixSaver('readPart', './private/models/lpReadInit/')
    dict1 = get_op('readPart')
    # print(dict1)
    saver1 = tf.train.Saver(dict1)
    # vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    # for var in vars:
    #     print(var.name)
    # saver = tf.train.Saver()
    #Optimize ONLY new vars.
    toOpt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='findPart')
    print('To Train')
    for v in toOpt:
        print(v.name)
    # optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss, var_list=toOpt)
    optimizer = tf.train.AdamOptimizer().minimize(loss, var_list=toOpt)
    # optimizer = tf.train.AdamOptimizer().minimize(loss)
    # pred = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqAfterConv, merge_repeated=False)[0][0])
    pred = tf.to_int32(ctc.ctc_greedy_decoder(logits3d, seqAfterConv)[0][0])
    edist = tf.edit_distance(pred, targetY, normalize=False)
    tgtLens = tf.to_float(tf.size(targetY.values))
    err = tf.reduce_sum(edist) / tgtLens
    saver = tf.train.Saver()

with tf.Session(graph=graph) as session:
    # writer = tf.train.SummaryWriter('./log', session.graph)
    print('Initializing')
    # tf.global_variables_initializer().run()

    saver1.restore(session, './private/models/lpReadInit/checkpoint-14')
    uVar = get_uninitialized_variables()
    tf.variables_initializer(var_list=uVar).run()
    uVarB = get_uninitialized_variables()
    for var in uVarB:
        print(var.name)

    # ckpt = tf.train.get_checkpoint_state("./private/models/lp_stn1/")
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(session, ckpt.model_checkpoint_path)
    #
    # print(ckpt)
    # workList = trainList5[:]
    # workList = trainList[:]
    # errV = 0
    # lossV = 0
    # timeVS = time.time()
    # for bStep in range(stepsPerEpocheVal):
    #     bList, workList = workList[:batchSize], workList[batchSize:]
    #     batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
    #                                                                                                        imgInW,
    #                                                                                                      mvn=False)
    #     feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
    #                 targetShape: batchTargetShape, keep_prob: 1.0, trainIN: False}
    #     lossB, aErr, p, s_o, f_a = session.run([loss, err, pred, stn_o, f_aff], feed_dict=feedDict)
    #     print(aErr)
    #     res = []
    #     for idx in p.values:
    #         res.append(cm.get_value(idx))
    #     print(res)
    #     print(f_a)
    #     # print(p)
    #     plt.imshow(s_o[0,:,:,0], cmap=plt.cm.gray)
    #     plt.show()
    #
    #     lossV += lossB
    #     errV += aErr
    # print('Val: CTC-loss ', lossV)
    # errVal = errV / stepsPerEpocheVal
    # print('Val: CER ', errVal)
    # print('Val time ', time.time() - timeVS)

    for epoch in range(nEpochs):
        if(epoch == 10):
            trainList.extend(trainList1)
        if (epoch == 30):
            trainList.extend(trainList2)
        if (epoch == 50):
            trainList.extend(trainList3)
        if (epoch == 70):
            trainList.extend(trainList4)
        if (epoch == 90):
            trainList.extend(trainList5)

        workList = trainList[:]
        shuffle(workList)
        workList = workList[0:numT]
        print('Epoch', epoch + 1, '...')
        lossT = 0
        errT = 0
        timeTS = time.time()
        for bStep in range(stepsPerEpocheTrain):
            bList, workList = workList[:batchSize], workList[batchSize:]
            batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
                                                                                                             imgInW,
                                                                                                             mvn=False)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, keep_prob: 0.5, trainIN: True}
            _, lossB, aErr = session.run([optimizer, loss, err], feed_dict=feedDict)
            # lossB, aErr = session.run([loss, err], feed_dict=feedDict)
            lossT += lossB
            errT += aErr
            # plt.imshow(s_o[0, :, :, 0], cmap=plt.cm.gray)
            # plt.show()
        print('Train: CTC-loss ', lossT)
        cerT = errT / stepsPerEpocheTrain
        print('Train: CER ', cerT)
        print('Train time ', time.time() - timeTS)
        workList = valList[:]
        errV = 0
        lossV = 0
        timeVS = time.time()
        for bStep in range(stepsPerEpocheVal):
            bList, workList = workList[:batchSize], workList[batchSize:]
            batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
                                                                                                             imgInW,
                                                                                                             mvn=False)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, keep_prob: 1.0, trainIN: False}
            lossB, aErr = session.run([loss, err], feed_dict=feedDict)
            # lossB, aErr, sE, sL = session.run([loss, err, err_val, loss_val], feed_dict=feedDict)
            # writer.add_summary(sE, epoch*stepsPerEpocheVal + bStep)
            # writer.add_summary(sL, epoch * stepsPerEpocheVal + bStep)
            lossV += lossB
            errV += aErr
        print('Val: CTC-loss ', lossV)
        errVal = errV / stepsPerEpocheVal
        print('Val: CER ', errVal)
        print('Val time ', time.time() - timeVS)
        # Write a checkpoint.
        checkpoint_file = os.path.join('./private/models/lp_stn1/', 'checkpoint')
        saver.save(session, checkpoint_file, global_step=epoch)

