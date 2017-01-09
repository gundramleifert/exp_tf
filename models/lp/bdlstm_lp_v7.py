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
        checkpoint_file = os.path.join('./private/models/lp7/', 'checkpoint')
        saver.save(session, checkpoint_file, global_step=epoch)

# Defining graph
# Initializing
# Epoch 1 ...
# Train: CTC-loss  48866.2581406
# Train: CER  0.965547385549
# Train time  1988.0135901
# Val: CTC-loss  3939.03445244
# Val: CER  0.922551070624
# Val time  76.6373090744
# Epoch 2 ...
# Train: CTC-loss  12164.0192926
# Train: CER  0.226216769428
# Train time  3142.16139221
# Val: CTC-loss  459.386833072
# Val: CER  0.0895408948057
# Val time  139.804624796
# Epoch 3 ...
# Train: CTC-loss  4646.65572649
# Train: CER  0.0825175480984
# Train time  3825.13034105
# Val: CTC-loss  351.43373245
# Val: CER  0.0676223002474
# Val time  138.911491871
# Epoch 4 ...
# Train: CTC-loss  3786.05023403
# Train: CER  0.0664457885989
# Train time  3796.00666809
# Val: CTC-loss  310.588638403
# Val: CER  0.0586238032516
# Val time  138.202473879
# Epoch 5 ...
# Train: CTC-loss  3323.13822353
# Train: CER  0.0585048161131
# Train time  3813.85453606
# Val: CTC-loss  293.197274931
# Val: CER  0.0550824288966
# Val time  133.58867383
# Epoch 6 ...
# Train: CTC-loss  3033.23137386
# Train: CER  0.0532705848029
# Train time  3781.45246887
# Val: CTC-loss  276.715799134
# Val: CER  0.0514238722881
# Val time  134.338634014
# Epoch 7 ...
# Train: CTC-loss  2814.65225264
# Train: CER  0.0497558537051
# Train time  3837.40190506
# Val: CTC-loss  271.768433571
# Val: CER  0.0507225987805
# Val time  138.49199605
# Epoch 8 ...
# Train: CTC-loss  2644.07710527
# Train: CER  0.0470508050831
# Train time  3759.10891795
# Val: CTC-loss  274.45960087
# Val: CER  0.0507211494284
# Val time  137.742328882
# Epoch 9 ...
# Train: CTC-loss  2482.13503463
# Train: CER  0.0443883210671
# Train time  3824.320997
# Val: CTC-loss  267.247882919
# Val: CER  0.0499490915952
# Val time  130.312234879
# Epoch 10 ...
# Train: CTC-loss  2352.3551532
# Train: CER  0.0422522745982
# Train time  3766.31166601
# Val: CTC-loss  273.866393549
# Val: CER  0.0512364817594
# Val time  131.097141027
# Epoch 11 ...
# Train: CTC-loss  2225.57266775
# Train: CER  0.0401326419915
# Train time  3810.56215405
# Val: CTC-loss  267.63567424
# Val: CER  0.0495922919363
# Val time  123.485896111
# Epoch 12 ...
# Train: CTC-loss  2112.3871738
# Train: CER  0.0381448065616
# Train time  3532.78759003
# Val: CTC-loss  270.081726674
# Val: CER  0.0495919786433
# Val time  119.069125175
# Epoch 13 ...
# Train: CTC-loss  1997.25157417
# Train: CER  0.0363146708998
# Train time  3323.93918204
# Val: CTC-loss  269.521347839
# Val: CER  0.0491224984952
# Val time  117.911461115
# Epoch 14 ...
# Train: CTC-loss  1898.6778805
# Train: CER  0.0346307474614
# Train time  3342.55619502
# Val: CTC-loss  284.072918201
# Val: CER  0.0506088375487
# Val time  120.113620996
# Epoch 15 ...
# Train: CTC-loss  1798.23522407
# Train: CER  0.0330799072123
# Train time  3309.24860597
# Val: CTC-loss  271.761080149
# Val: CER  0.0485061887542
# Val time  120.585114002