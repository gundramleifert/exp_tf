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
from util.CharacterMapper import get_cm_lp
from util.saver import PrefixSaver
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

def get_saver_dict(prefix):
    dict = {}
    if prefix[-1] != '/':
        prefix = prefix + '/'
    res = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix)
    for t in res:
        key = t.name
        key = key[len(prefix):]
        dict[str(key)] = t
        # print(dict)
    return dict

def inference(images, seqLen, keep_prob, phase_train):
    with tf.variable_scope('readPart') as scope:
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
                        targetShape: batchTargetShape, seqLengths: imgW*np.ones([batchSize]), keep_prob: 0.5, trainIN: True}
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
                        targetShape: batchTargetShape, seqLengths: imgW*np.ones([batchSize]), keep_prob: 1.0, trainIN: False}
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
# Train: CTC-loss  425823.728893
# Train: CER  0.53363006419
# Train time  1443.70941591
# Val: CTC-loss  7984.01521397
# Val: CER  0.104459231919
# Val time  34.3682131767
# Epoch 2 ...
# Train: CTC-loss  64123.7205703
# Train: CER  0.0713416529448
# Train time  1377.6435709
# Val: CTC-loss  4989.27085871
# Val: CER  0.0612603672307
# Val time  30.7804141045
# Epoch 3 ...
# Train: CTC-loss  52995.6194758
# Train: CER  0.0583216237314
# Train time  1350.16746306
# Val: CTC-loss  4974.11897206
# Val: CER  0.0643494823141
# Val time  30.9492008686
# Epoch 4 ...
# Train: CTC-loss  47650.4380932
# Train: CER  0.0526525123744
# Train time  1352.0392859
# Val: CTC-loss  3726.932432
# Val: CER  0.0427109787071
# Val time  31.1687119007
# Epoch 5 ...
# Train: CTC-loss  42036.9393945
# Train: CER  0.0459474870026
# Train time  1310.07066607
# Val: CTC-loss  3805.60591072
# Val: CER  0.0458993885049
# Val time  30.856719017
# Epoch 6 ...
# Train: CTC-loss  39825.6757735
# Train: CER  0.0435396286173
# Train time  1314.13341784
# Val: CTC-loss  3652.99790461
# Val: CER  0.0441121808328
# Val time  31.0514140129
# Epoch 7 ...
# Train: CTC-loss  38805.7285916
# Train: CER  0.0428731738658
# Train time  1302.10318995
# Val: CTC-loss  3396.00391503
# Val: CER  0.0390877672894
# Val time  31.3051400185
# Epoch 8 ...
# Train: CTC-loss  37655.5376481
# Train: CER  0.040996678151
# Train time  1272.14776897
# Val: CTC-loss  3241.93075249
# Val: CER  0.0379257008961
# Val time  31.1723740101
# Epoch 9 ...
# Train: CTC-loss  35306.6837268
# Train: CER  0.039340560049
# Train time  1255.14130592
# Val: CTC-loss  3298.8084621
# Val: CER  0.0380707961522
# Val time  30.9955301285
# Epoch 10 ...
# Train: CTC-loss  33064.6664063
# Train: CER  0.0367346494644
# Train time  1257.81966996
# Val: CTC-loss  3216.98754848
# Val: CER  0.0373557400317
# Val time  31.348790884
# Epoch 11 ...
# Train: CTC-loss  33231.565251
# Train: CER  0.0367647848037
# Train time  1223.39252186
# Val: CTC-loss  3369.77349287
# Val: CER  0.0383404844196
# Val time  31.4620189667
# Epoch 12 ...
# Train: CTC-loss  31701.1454951
# Train: CER  0.0351086833043
# Train time  1228.37678313
# Val: CTC-loss  3071.02720545
# Val: CER  0.0351393960496
# Val time  20.628880024
# Epoch 13 ...
# Train: CTC-loss  32437.9929693
# Train: CER  0.0362730159413
# Train time  1223.4059422
# Val: CTC-loss  3160.40971142
# Val: CER  0.0356085528445
# Val time  31.3786180019
# Epoch 14 ...
# Train: CTC-loss  30465.1829899
# Train: CER  0.034653937174
# Train time  1197.5064919
# Val: CTC-loss  3055.63726359
# Val: CER  0.0346931330959
# Val time  31.2907111645
# Epoch 15 ...
# Train: CTC-loss  29634.7127888
# Train: CER  0.0332576885411
# Train time  1208.09267807
# Val: CTC-loss  3150.66269298
# Val: CER  0.0346802188741
# Val time  31.6104729176