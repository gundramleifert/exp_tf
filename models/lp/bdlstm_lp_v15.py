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


# def batch_norm(x, n_out, phase_train):
#     """
#     Batch normalization on convolutional maps.
#     Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
#     Args:
#         x:           Tensor, 4D BHWD input maps
#         n_out:       integer, depth of input maps
#         phase_train: boolean tf.Varialbe, true indicates training phase
#         scope:       string, variable scope
#     Return:
#         normed:      batch-normalized maps
#     """
#     with tf.variable_scope('bn'):
#         beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
#                                      name='beta', trainable=True)
#         gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
#                                       name='gamma', trainable=True)
#         batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
#         ema = tf.train.ExponentialMovingAverage(decay=0.99)
#
#         def mean_var_with_update():
#             ema_apply_op = ema.apply([batch_mean, batch_var])
#             with tf.control_dependencies([ema_apply_op]):
#                 return tf.identity(batch_mean), tf.identity(batch_var)
#
#         mean, var = tf.cond(phase_train,
#                             mean_var_with_update,
#                             lambda: (ema.average(batch_mean), ema.average(batch_var)))
#         normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
#     return normed

def inference(images, seqLen, phase_train):
    with tf.variable_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([6, 5, channels, 32], stddev=5e-2), name='weights')
        ##Weight Decay?
        # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
        # tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(images, kernel, [1, 4, 3, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.1, shape=[32]), name='biases')
        pre_activation = tf.nn.bias_add(conv, biases)
        # conv1_bn = batch_norm(pre_activation, decay= 0.99, is_training=phase_train, scope=scope, outputs_collections=None)
        conv1_bn = batch_norm(pre_activation, decay= 0.999, is_training=phase_train, scope="BN1")
        conv1 = tf.nn.relu(conv1_bn, name=scope.name)
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
        # conv2_bn = batch_norm(pre_activation, decay= 0.99, is_training=phase_train, scope=scope, outputs_collections=None)
        conv2_bn = batch_norm(pre_activation, decay= 0.999, is_training=phase_train, scope="BN2")
        conv2 = tf.nn.relu(conv2_bn, name=scope.name)
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
        # conv3_bn = batch_norm(pre_activation, decay= 0.99, is_training=phase_train, scope=scope, outputs_collections=None)
        conv3_bn = batch_norm(pre_activation, decay= 0.999, is_training=phase_train, scope="BN3")
        conv3 = tf.nn.relu(conv3_bn, name=scope.name)
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
    trainIN = tf.placeholder_with_default(tf.constant(False),[])
    # trainIN = tf.placeholder(tf.bool)
    logits3d, seqAfterConv = inference(inputX, seqLengths, trainIN)
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
    # There are pathes control by a bool.... so we have to init all variables CAREFULLY
    # vars_to_train = tf.trainable_variables()
    # vars_for_bn1 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv_1/bn')
    # vars_for_bn2 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv_2/bn')
    # vars_for_bn3 = tf.get_collection(tf.GraphKeys.VARIABLES, scope='conv_2/bn')
    # vars_to_train = list(set(vars_to_train).union(set(vars_for_bn1)))
    # vars_to_train = list(set(vars_to_train).union(set(vars_for_bn2)))
    # vars_to_train = list(set(vars_to_train).union(set(vars_for_bn3)))

    # init = tf.variables_initializer(vars_to_train)
    init = tf.global_variables_initializer()
    # feedDictInit = {trainIN: True}
    # session.run(init, feedDictInit)
    # feedDictInit = {trainIN: False}
    # session.run(init, feedDictInit)
    session.run(init)
    #
    # tf.global_variables_initializer().run()
    #
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
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths, trainIN: True}
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
            trainV = False
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, seqLengths: batchSeqLengths, trainIN: False}
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
        checkpoint_file = os.path.join('./private/models/lp15/', 'checkpoint')
        saver.save(session, checkpoint_file, global_step=epoch)

# Defining graph
# Initializing
# Epoch 1 ...
# Train: CTC-loss  201805.916738
# Train: CER  0.239930366529
# Train time  1329.62995696
# Val: CTC-loss  6184.65547276
# Val: CER  0.0755592971482
# Val time  31.4458889961
# Epoch 2 ...
# Train: CTC-loss  52119.7021423
# Train: CER  0.0574641779402
# Train time  928.213298082
# Val: CTC-loss  4162.47086686
# Val: CER  0.0503549973187
# Val time  31.220541954
# Epoch 3 ...
# Train: CTC-loss  44906.9371882
# Train: CER  0.0501647832312
# Train time  915.68283987
# Val: CTC-loss  3995.60864985
# Val: CER  0.046805318948
# Val time  31.3634371758
# Epoch 4 ...
# Train: CTC-loss  40642.2453663
# Train: CER  0.0459106236174
# Train time  932.547461987
# Val: CTC-loss  3806.43163115
# Val: CER  0.0464194491356
# Val time  29.1131699085
# Epoch 5 ...
# Train: CTC-loss  37478.2149878
# Train: CER  0.042565695427
# Train time  2095.75372505
# Val: CTC-loss  3719.89040187
# Val: CER  0.0432963068323
# Val time  85.8479371071
# Epoch 6 ...
# Train: CTC-loss  35043.3060023
# Train: CER  0.0402104348034
# Train time  2132.17571497
# Val: CTC-loss  3707.31053054
# Val: CER  0.0433541805668
# Val time  70.6841762066
# Epoch 7 ...
# Train: CTC-loss  32464.6306195
# Train: CER  0.037253022357
# Train time  2063.37620187
# Val: CTC-loss  3881.23983011
# Val: CER  0.0443246099939
# Val time  70.9693470001
# Epoch 8 ...
# Train: CTC-loss  31083.6049052
# Train: CER  0.0356593503062
# Train time  2069.09882188
# Val: CTC-loss  3598.8268586
# Val: CER  0.0396351277938
# Val time  70.4906511307
# Epoch 9 ...
# Train: CTC-loss  30087.5940406
# Train: CER  0.0350050771687
# Train time  2121.05450296
# Val: CTC-loss  3545.43416484
# Val: CER  0.0397489090415
# Val time  65.8404579163
# Epoch 10 ...
# Train: CTC-loss  27056.3291208
# Train: CER  0.0319132109034
# Train time  2151.9467299
# Val: CTC-loss  3707.16886334
# Val: CER  0.041531600327
# Val time  75.3269109726
# Epoch 11 ...
# Train: CTC-loss  25898.2382009
# Train: CER  0.0309886606487
# Train time  2161.93321013
# Val: CTC-loss  3722.8062934
# Val: CER  0.0400601589534
# Val time  73.3718810081
# Epoch 12 ...
# Train: CTC-loss  25070.5427481
# Train: CER  0.0304743385093
# Train time  2182.19665098
# Val: CTC-loss  3859.30628511
# Val: CER  0.0415965017519
# Val time  66.9144608974
# Epoch 13 ...
# Train: CTC-loss  23551.1893152
# Train: CER  0.028695252275
# Train time  2201.50485516
# Val: CTC-loss  3969.43364912
# Val: CER  0.0421210426917
# Val time  76.5628390312
# Epoch 14 ...
# Train: CTC-loss  22448.6657824
# Train: CER  0.0277596670338
# Train time  2197.82337308
# Val: CTC-loss  3902.78446943
# Val: CER  0.0404049656528
# Val time  76.5496060848
# Epoch 15 ...
# Train: CTC-loss  21249.4980625
# Train: CER  0.0264487374627
# Train time  2207.32114792
# Val: CTC-loss  3958.85068892
# Val: CER  0.0404411036282
# Val time  75.936175108