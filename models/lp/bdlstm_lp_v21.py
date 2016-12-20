'''

Author: Tobi and Gundram
'''

from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.contrib.layers import batch_norm
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops.rnn import bidirectional_rnn
from util.LoaderUtil import read_image_list, get_list_vals
from util.CharacterMapper import get_cm_lp
from util.saver import PrefixSaver
from random import shuffle
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Goes done to 10%
INPUT_PATH_TRAIN = './private/lists/lp_only_shifted_train.lst'
INPUT_PATH_VAL = './private/lists/lp_only_val.lst'
cm = get_cm_lp()
# Additional NaC Channel
nClasses = cm.size() + 1

nEpochs = 15
batchSize = 16
# learningRate = 0.001
# momentum = 0.9
# It is assumed that the TextLines are ALL saved with a consistent height of imgH
imgH = 48
# Depending on the size the image is cropped or zero padded
imgW = 256
channels = 1
nHiddenLSTM1 = 256


os.chdir("../..")
trainList = read_image_list(INPUT_PATH_TRAIN)
numT = 32998
stepsPerEpocheTrain = numT / batchSize
valList = read_image_list(INPUT_PATH_VAL)
stepsPerEpocheVal = len(valList) / batchSize


def inference(images, seqLen, keep_prob, phase_train):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([6, 5, channels, 32], stddev=5e-2), name='weights')
        ##Weight Decay?
        # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(images, kernel, [1, 4, 3, 1], padding='SAME')
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
    inputX = tf.placeholder(tf.float32, shape=(batchSize, imgH, imgW, channels))
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))
    keep_prob = tf.placeholder(tf.float32)
    trainIN = tf.placeholder_with_default(tf.constant(False), [])
    logits3d, seqAfterConv = inference(inputX, seqLengths, keep_prob, trainIN)
    loss = loss(logits3d, targetY, seqAfterConv)
    saver = PrefixSaver('readPart', './private/models/lp23/')
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

    # ckpt = tf.train.get_checkpoint_state("./private/models/lp2/")
    # if ckpt and ckpt.model_checkpoint_path:
    #     saver.restore(session, ckpt.model_checkpoint_path)
    # print(ckpt)
    # workList = valList[:]
    # errV = 0
    # lossV = 0
    # timeVS = time.time()
    # cmInv = get_charmap_lp_inv()
    # for bStep in range(stepsPerEpocheVal):
    #     bList, workList = workList[:batchSize], workList[batchSize:]
    #     batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
    #                                                                                                        imgW,
    #                                                                                                      mvn=True)
    #     feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
    #                 targetShape: batchTargetShape, seqLengths: batchSeqLengths}
    #     lossB, aErr, p = session.run([loss, err, pred], feed_dict=feedDict)
    #     print(aErr)
    #     res = []
    #     for idx in p.values:
    #         res.append(cmInv[idx])
    #     print(res)
    #     # print(p)
    #     plt.imshow(batchInputs[0,:,:,0], cmap=plt.cm.gray)
    #     plt.show()
    #
    #     lossV += lossB
    #     errV += aErr
    # print('Val: CTC-loss ', lossV)
    # errVal = errV / stepsPerEpocheVal
    # print('Val: CER ', errVal)
    # print('Val time ', time.time() - timeVS)
    for epoch in range(nEpochs):
        workList = trainList[:]
        shuffle(workList)
        workList = workList[0:32998]
        print('Epoch', epoch + 1, '...')
        lossT = 0
        errT = 0
        timeTS = time.time()
        for bStep in range(stepsPerEpocheTrain):
            bList, workList = workList[:batchSize], workList[batchSize:]
            batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
                                                                                                             imgW,
                                                                                                             mvn=True)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths, keep_prob: 0.5, trainIN: True}
            _, lossB, aErr = session.run([optimizer, loss, err], feed_dict=feedDict)
            # _, lossB, aErr, sET, sLT = session.run([optimizer, loss, err, err_train, loss_train], feed_dict=feedDict)
            lossT += lossB
            # writer.add_summary(sET, epoch * stepsPerEpocheTrain + bStep)
            # writer.add_summary(sLT, epoch * stepsPerEpocheTrain + bStep)
            errT += aErr
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
                                                                                                             imgW,
                                                                                                             mvn=True)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths, keep_prob: 1.0, trainIN: False}
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
        saver.save(session, global_step=epoch)

# Defining graph
# Initializing
# Epoch 1 ...
# Train: CTC-loss  297550.189748
# Train: CER  0.369817732036
# Train time  656.495547771
# Val: CTC-loss  6243.76747084
# Val: CER  0.0772731669476
# Val time  19.8217790127
# Epoch 2 ...
# Train: CTC-loss  61101.7735276
# Train: CER  0.0697167958616
# Train time  645.201889038
# Val: CTC-loss  4302.78418201
# Val: CER  0.0530221136038
# Val time  19.2986240387
# Epoch 3 ...
# Train: CTC-loss  52259.9170997
# Train: CER  0.0591758824659
# Train time  644.602695942
# Val: CTC-loss  4115.12788808
# Val: CER  0.0497590430519
# Val time  19.3517248631
# Epoch 4 ...
# Train: CTC-loss  45914.7382628
# Train: CER  0.05329392137
# Train time  647.244607925
# Val: CTC-loss  3893.34798646
# Val: CER  0.0468291321241
# Val time  19.2186748981
# Epoch 5 ...
# Train: CTC-loss  46103.7913166
# Train: CER  0.0531511599456
# Train time  641.510627985
# Val: CTC-loss  3688.32746397
# Val: CER  0.0446895566435
# Val time  19.180918932
# Epoch 6 ...
# Train: CTC-loss  42205.5880004
# Train: CER  0.048825094125
# Train time  642.546453953
# Val: CTC-loss  4003.83163607
# Val: CER  0.0485505199578
# Val time  19.301680088
# Epoch 7 ...
# Train: CTC-loss  39880.1079626
# Train: CER  0.0460336496633
# Train time  642.13598609
# Val: CTC-loss  3590.54607366
# Val: CER  0.044335270928
# Val time  19.2930381298
# Epoch 8 ...
# Train: CTC-loss  38284.5067011
# Train: CER  0.0447305060027
# Train time  642.322103977
# Val: CTC-loss  3573.68837625
# Val: CER  0.0430186986824
# Val time  19.3296511173
# Epoch 9 ...
# Train: CTC-loss  38405.1099423
# Train: CER  0.0448783215015
# Train time  641.295161009
# Val: CTC-loss  3487.68989562
# Val: CER  0.0414736744734
# Val time  19.2940099239
# Epoch 10 ...
# Train: CTC-loss  36982.8467782
# Train: CER  0.0429310147844
# Train time  641.01845789
# Val: CTC-loss  3327.72588518
# Val: CER  0.0399345461638
# Val time  19.2967851162
# Epoch 11 ...
# Train: CTC-loss  34145.1438549
# Train: CER  0.0405711921254
# Train time  641.68900013
# Val: CTC-loss  3375.23555307
# Val: CER  0.0392742058173
# Val time  19.3452589512
# Epoch 12 ...
# Train: CTC-loss  34995.7241417
# Train: CER  0.0409852813854
# Train time  642.184346914
# Val: CTC-loss  3246.25002918
# Val: CER  0.0378713623775
# Val time  19.2844979763
# Epoch 13 ...
# Train: CTC-loss  34225.7201001
# Train: CER  0.0391374439716
# Train time  641.09472394
# Val: CTC-loss  3408.6861199
# Val: CER  0.040287604416
# Val time  19.2703030109
# Epoch 14 ...
# Train: CTC-loss  33689.406593
# Train: CER  0.0391193181349
# Train time  640.176419973
# Val: CTC-loss  3340.19572805
# Val: CER  0.0382310437377
# Val time  19.2953748703
# Epoch 15 ...
# Train: CTC-loss  32967.1510995
# Train: CER  0.0391116109469
# Train time  643.672610998
# Val: CTC-loss  3323.21851535
# Val: CER  0.0397801826295
# Val time  19.3347668648