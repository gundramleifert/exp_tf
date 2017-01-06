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
batchSize = 4
learningRate = 0.001
momentum = 0.9
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


def inference(images, seqLen):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([6, 5, channels, 32], stddev=5e-2), name='weights')
        ##Weight Decay?
        # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(images, kernel, [1, 4, 3, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[32]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        # _activation_summary(conv1)
        # norm1 = tf.nn.local_response_normalization(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1')
        seqFloat = tf.to_float(seqLen)
        seqL2 = tf.ceil(seqFloat * 0.33)
    with tf.variable_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=5e-2), name='weights')
        ##Weight Decay?
        # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(conv1, kernel, [1, 1, 1, 1], padding='SAME')
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
        backwardH1 = rnn_cell.LSTMCell(nHiddenLSTM1, use_peepholes=True, state_is_tuple=True)
        outputs, _, _ = bidirectional_rnn(forwardH1, backwardH1, rnnIn, dtype=tf.float32)
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
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
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
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths}
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
        checkpoint_file = os.path.join('./private/models/lp3/', 'checkpoint')
        saver.save(session, checkpoint_file, global_step=epoch)
# Defining graph
# Initializing
# Epoch 1 ...
# Train: CTC-loss  127770.795485
# Train: CER  0.632449947914
# Train time  1623.64185095
# Val: CTC-loss  1690.94079254
# Val: CER  0.0842024714947
# Val time  53.6754989624
# Epoch 2 ...
# Train: CTC-loss  16142.664189
# Train: CER  0.0719488524786
# Train time  1643.34809399
# Val: CTC-loss  1180.64658372
# Val: CER  0.0566876666149
# Val time  52.8805279732
# Epoch 3 ...
# Train: CTC-loss  12782.7443373
# Train: CER  0.0572215012803
# Train time  1642.10430193
# Val: CTC-loss  1087.44890879
# Val: CER  0.0513329066634
# Val time  53.3245558739
# Epoch 4 ...
# Train: CTC-loss  11220.8490879
# Train: CER  0.0502287249668
# Train time  1634.754421
# Val: CTC-loss  1082.34532301
# Val: CER  0.0496852177829
# Val time  52.6013569832
# Epoch 5 ...
# Train: CTC-loss  10291.8197946
# Train: CER  0.0467629148429
# Train time  1625.04631186
# Val: CTC-loss  1040.13378663
# Val: CER  0.0486953979631
# Val time  52.304046154
# Epoch 6 ...
# Train: CTC-loss  9489.53495804
# Train: CER  0.0432804138749
# Train time  1622.64882994
# Val: CTC-loss  1023.34265649
# Val: CER  0.0468253398041
# Val time  52.2032320499
# Epoch 7 ...
# Train: CTC-loss  8878.37165775
# Train: CER  0.0410694975078
# Train time  1616.60257196
# Val: CTC-loss  998.865922881
# Val: CER  0.0455678806404
# Val time  51.6127068996
# Epoch 8 ...
# Train: CTC-loss  8349.18803357
# Train: CER  0.0387621530634
# Train time  1613.39381695
# Val: CTC-loss  1005.80148646
# Val: CER  0.0450465412537
# Val time  51.599864006
# Epoch 9 ...
# Train: CTC-loss  7863.9768993
# Train: CER  0.0373975759733
# Train time  2364.71975899
# Val: CTC-loss  1025.37011006
# Val: CER  0.0448487868408
# Val time  127.908433914
# Epoch 10 ...
# Train: CTC-loss  7474.0068357
# Train: CER  0.0360007262832
# Train time  5664.99592304
# Val: CTC-loss  1030.9719641
# Val: CER  0.045286204497
# Val time  240.020387888
# Epoch 11 ...
# Train: CTC-loss  7192.40476506
# Train: CER  0.0348674249148
# Train time  7002.88452506
# Val: CTC-loss  1045.4683604
# Val: CER  0.0458950943997
# Val time  232.987906933
# Epoch 12 ...
# Train: CTC-loss  6885.85843309
# Train: CER  0.0336379728044
# Train time  6940.85707903
# Val: CTC-loss  1063.69808645
# Val: CER  0.0471115465313
# Val time  226.829921961
# Epoch 13 ...
# Train: CTC-loss  6587.33486168
# Train: CER  0.0324386308837
# Train time  7016.62783194
# Val: CTC-loss  1115.10326525
# Val: CER  0.0477711562912
# Val time  233.850280046
# Epoch 14 ...
# Train: CTC-loss  6396.92553726
# Train: CER  0.0316869818918
# Train time  6944.38106823
# Val: CTC-loss  1161.44386697
# Val: CER  0.0474798077444
# Val time  237.52304101
# Epoch 15 ...
# Train: CTC-loss  6218.94439296
# Train: CER  0.0306587755172
# Train time  6963.23701501
# Val: CTC-loss  1119.81188818
# Val: CER  0.0477910614709
# Val time  235.797486067