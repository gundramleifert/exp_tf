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
stepsPerEpocheTrain = len(trainList) / batchSize
valList = read_image_list(INPUT_PATH_VAL)
stepsPerEpocheVal = len(valList) / batchSize


def inference(images, seqLen, keep_prob):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([6, 5, channels, 32], stddev=5e-2), name='weights')
        ##Weight Decay?
        # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(images, kernel, [1, 4, 3, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[32]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
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
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        # _activation_summary(conv2)
        # norm2
        # norm2 = tf.nn.local_response_normalization(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm2')
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 4, 2, 1], strides=[1, 4, 2, 1], padding='SAME', name='pool2')
        seqL3 = tf.ceil(seqL2 * 0.5)
    with tf.variable_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 3, 64, 128], stddev=5e-2), name='weights')
        ##Weight Decay?
        # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[128]), name='biases')
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
    keep_prob = tf.placeholder(tf.float32)
    logits3d, seqAfterConv = inference(inputX, seqLengths, keep_prob)
    loss = loss(logits3d, targetY, seqAfterConv)
    # optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)
    optimizer = tf.train.AdamOptimizer().minimize(loss)
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
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths, keep_prob: 0.5}
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
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths, keep_prob: 1.0}
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
        checkpoint_file = os.path.join('./private/models/lp14/', 'checkpoint')
        saver.save(session, checkpoint_file, global_step=epoch)
#
# Defining graph
# Initializing
# Epoch 1 ...
# Train: CTC-loss  28508.7277168
# Train: CER  0.570930393472
# Train time  526.916748047
# Val: CTC-loss  347.704063319
# Val: CER  0.0707173571937
# Val time  16.5800960064
# Epoch 2 ...
# Train: CTC-loss  3773.3409663
# Train: CER  0.0687460959049
# Train time  522.388767004
# Val: CTC-loss  272.768027304
# Val: CER  0.0547157528257
# Val time  17.6897339821
# Epoch 3 ...
# Train: CTC-loss  3059.43834222
# Train: CER  0.0557980481151
# Train time  523.647897959
# Val: CTC-loss  252.038394164
# Val: CER  0.0491571270806
# Val time  16.3676099777
# Epoch 4 ...
# Train: CTC-loss  2770.29840848
# Train: CER  0.0505689580358
# Train time  514.358065128
# Val: CTC-loss  222.682988223
# Val: CER  0.0438422592765
# Val time  16.28429389
# Epoch 5 ...
# Train: CTC-loss  2556.53836938
# Train: CER  0.0472107773473
# Train time  510.437147856
# Val: CTC-loss  224.31911112
# Val: CER  0.0437619703608
# Val time  16.2737751007
# Epoch 6 ...
# Train: CTC-loss  2417.34813369
# Train: CER  0.0448309326423
# Train time  510.649667025
# Val: CTC-loss  219.999615309
# Val: CER  0.0416769259784
# Val time  16.4510471821
# Epoch 7 ...
# Train: CTC-loss  2254.09395537
# Train: CER  0.0425710683823
# Train time  523.477609873
# Val: CTC-loss  219.126966105
# Val: CER  0.0415681593748
# Val time  16.2184100151
# Epoch 8 ...
# Train: CTC-loss  2167.10055237
# Train: CER  0.0413632454932
# Train time  519.34996295
# Val: CTC-loss  218.289330068
# Val: CER  0.0413328826198
# Val time  16.4323670864
# Epoch 9 ...
# Train: CTC-loss  2108.66918822
# Train: CER  0.0406193800892
# Train time  612.268445015
# Val: CTC-loss  233.110707047
# Val: CER  0.0422292060772
# Val time  27.7160820961
# Epoch 10 ...
# Train: CTC-loss  1994.81178643
# Train: CER  0.0386657376207
# Train time  777.933701992
# Val: CTC-loss  235.526006107
# Val: CER  0.0443903178594
# Val time  16.2069129944
# Epoch 11 ...
# Train: CTC-loss  1927.48697802
# Train: CER  0.0378296070309
# Train time  507.605500937
# Val: CTC-loss  248.664825527
# Val: CER  0.0446493811457
# Val time  16.2018079758
# Epoch 12 ...
# Train: CTC-loss  1868.92664417
# Train: CER  0.036412956395
# Train time  505.235450983
# Val: CTC-loss  230.760530531
# Val: CER  0.0402729476648
# Val time  16.2109811306
# Epoch 13 ...
# Train: CTC-loss  1780.21256075
# Train: CER  0.0352742843593
# Train time  505.75813508
# Val: CTC-loss  241.953211267
# Val: CER  0.0410179895832
# Val time  16.1639330387
# Epoch 14 ...
# Train: CTC-loss  1751.09394549
# Train: CER  0.0347554176982
# Train time  550.651408911
# Val: CTC-loss  239.090584156
# Val: CER  0.0415688001845
# Val time  27.9084730148
# Epoch 15 ...
# Train: CTC-loss  1694.777737
# Train: CER  0.0334086555759
# Train time  841.032886982
# Val: CTC-loss  245.870335489
# Val: CER  0.0409362354132
# Val time  25.564193964
