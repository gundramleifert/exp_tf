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
INPUT_PATH_TRAIN = './private/lists/lp_only_shifted_train.lst'
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
numT = 32998
stepsPerEpocheTrain = numT / batchSize
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
        checkpoint_file = os.path.join('./private/models/lp17/', 'checkpoint')
        saver.save(session, checkpoint_file, global_step=epoch)

# Defining graph
# Initializing
# Epoch 1 ...
# Train: CTC-loss  27943.1542435
# Train: CER  0.559750040776
# Train time  2191.12088585
# Val: CTC-loss  327.347985655
# Val: CER  0.06473718098
# Val time  79.4438340664
# Epoch 2 ...
# Train: CTC-loss  3656.58357514
# Train: CER  0.0670623835912
# Train time  2169.04920101
# Val: CTC-loss  262.219634719
# Val: CER  0.0500296886642
# Val time  65.3584258556
# Epoch 3 ...
# Train: CTC-loss  3031.79102664
# Train: CER  0.0555812349691
# Train time  2085.7185421
# Val: CTC-loss  241.713324932
# Val: CER  0.0478548960422
# Val time  66.4537339211
# Epoch 4 ...
# Train: CTC-loss  2837.13362656
# Train: CER  0.0526895946724
# Train time  2088.69855499
# Val: CTC-loss  226.507058132
# Val: CER  0.0419726519332
# Val time  66.8201479912
# Epoch 5 ...
# Train: CTC-loss  2735.5338171
# Train: CER  0.05055965747
# Train time  2058.14341998
# Val: CTC-loss  225.463234632
# Val: CER  0.0434549365013
# Val time  67.925606966
# Epoch 6 ...
# Train: CTC-loss  2515.81299276
# Train: CER  0.0467301957561
# Train time  2047.57153487
# Val: CTC-loss  238.905577891
# Val: CER  0.0456867918253
# Val time  59.0259740353
# Epoch 7 ...
# Train: CTC-loss  2472.89151211
# Train: CER  0.0459293686937
# Train time  2043.62233996
# Val: CTC-loss  214.839056365
# Val: CER  0.0401932810169
# Val time  65.5044369698
# Epoch 8 ...
# Train: CTC-loss  2319.6201498
# Train: CER  0.0431482211712
# Train time  2045.51919913
# Val: CTC-loss  220.818361165
# Val: CER  0.0404066477459
# Val time  68.7823340893
# Epoch 9 ...
# Train: CTC-loss  2262.53617008
# Train: CER  0.042402244173
# Train time  2016.28884506
# Val: CTC-loss  214.452787713
# Val: CER  0.0394926889287
# Val time  69.2747659683
# Epoch 10 ...
# Train: CTC-loss  2231.38164497
# Train: CER  0.0417519260693
# Train time  2024.70146608
# Val: CTC-loss  222.545890113
# Val: CER  0.0406456459563
# Val time  68.9941468239
# Epoch 11 ...
# Train: CTC-loss  2190.00648982
# Train: CER  0.0415818032874
# Train time  2018.26658416
# Val: CTC-loss  217.842715837
# Val: CER  0.040246285528
# Val time  69.1917479038
# Epoch 12 ...
# Train: CTC-loss  2173.41169817
# Train: CER  0.0405062055017
# Train time  1750.08354807
# Val: CTC-loss  221.152309516
# Val: CER  0.039514982789
# Val time  56.6253108978
# Epoch 13 ...
# Train: CTC-loss  2043.3760559
# Train: CER  0.0383939914366
# Train time  1639.2343328
# Val: CTC-loss  220.073389142
# Val: CER  0.0391547077878
# Val time  56.106842041
# Epoch 14 ...
# Train: CTC-loss  2085.97750598
# Train: CER  0.038905480022
# Train time  1628.85338998
# Val: CTC-loss  218.08703786
# Val: CER  0.0397401642363
# Val time  55.8922288418
# Epoch 15 ...
# Train: CTC-loss  2034.2408982
# Train: CER  0.0382120917112
# Train time  1628.99455309
# Val: CTC-loss  225.735546921
# Val: CER  0.0414734927061
# Val time  56.0414531231