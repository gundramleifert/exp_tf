'''

Author: Tobi and Gundram
'''

from __future__ import print_function


import tensorflow as tf
from numpy.lib.function_base import average

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

nEpochs = 150
batchSize = 16
sampleNum = 10

samplePerUpdate = batchSize*sampleNum
# Goes done to 10%
INPUT_PATH_TRAIN = './private/lists/lp0.lst'
INPUT_PATH_TRAIN1 = './private/lists/lp1.lst'
INPUT_PATH_TRAIN2 = './private/lists/lp2.lst'
INPUT_PATH_TRAIN3 = './private/lists/lp3.lst'
INPUT_PATH_TRAIN4 = './private/lists/lp4.lst'
INPUT_PATH_TRAIN5 = './private/lists/lp_enlarge_train.lst'
# INPUT_PATH_VAL = './private/lists/lp0.lst'
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
numT = 6400
stepsPerEpocheTrain = numT / batchSize
valList = read_image_list(INPUT_PATH_VAL)
valList = valList[:numT]
stepsPerEpocheVal = len(valList) / samplePerUpdate

channels = 1
#####This is the image size, for the READ part of the net.... so the output of the last STN
# It is assumed that the TextLines are ALL saved with a consistent height of imgH
imgH = 48
# Depending on the size the image is cropped or zero padded
imgW = 256
nHiddenLSTM1 = 256

imgInW = 512
imgInH = 512

loc_std = 0.22
grad_norm=4.0


def inference(images, targetY, seqLen, phase_train):
    with tf.variable_scope('findPart') as scope:
        imagesRes = tf.image.resize_bilinear(images, (128,128))
        # imagesRes = tf.image.resize_bilinear(images, (64,64))
        with tf.variable_scope('ff1') as scope:
            W_fc_loc1 = tf.Variable(tf.truncated_normal([16384, 20], stddev=5e-3), name='weights_loc1')
            # W_fc_loc1 = tf.Variable(tf.truncated_normal([4096, 20], stddev=5e-2), name='weights_loc1')
            b_fc_loc1 = tf.Variable(tf.truncated_normal([20], stddev=5e-3), name='bias_loc1')
            W_fc_loc2 = tf.Variable(tf.truncated_normal([20, 2], stddev=5e-3), name='weights_loc2')
            # Use identity transformation as starting point
            t_x = tf.Variable(0.0, name='t_x',dtype='float32')
            t_y = tf.Variable(0.0, name='t_y',dtype='float32')
            b_fc_loc2 = [t_x, t_y]
            # %% Define the two layer localisation network
            # inp_flat = tf.reshape(imagesRes, [samplePerUpdate, 4096])
            inp_flat = tf.reshape(imagesRes, [samplePerUpdate, 16384])
            h_fc_loc1 = tf.nn.tanh(tf.matmul(inp_flat, W_fc_loc1) + b_fc_loc1)
            # %% Second layer
            h_fc_loc2 = tf.matmul(h_fc_loc1, W_fc_loc2) + b_fc_loc2

            loc_mean = tf.clip_by_value(h_fc_loc2, -1., 1.)
            # loc_sample = loc_mean + tf.random_normal((samplePerUpdate, 2), stddev=loc_std)
            loc_sample = tf.cond(phase_train,lambda: loc_mean + tf.random_normal((samplePerUpdate, 2), stddev=loc_std),lambda: loc_mean)
            # loc_sample = tf.add(loc_mean,tf.random_normal((samplePerUpdate, 2), stddev=loc_std))
            loc_sample = tf.clip_by_value(loc_sample, -1.,1.)
            loc_sample = tf.stop_gradient(loc_sample)

            # loc_mean = tf.stop_gradient(loc_mean)

            # print(h_fc_loc2[:,0].get_shape())
            # print(tf.constant(0,dtype='float32', shape=[batchSize]).get_shape())

            s_x = tf.constant(0.5,dtype='float32', shape=[samplePerUpdate], name = 's_x')
            s_y = tf.constant(0.093,dtype='float32', shape=[samplePerUpdate], name = 's_y')
            r_1=tf.constant(0,dtype='float32', shape=[samplePerUpdate])
            r_2=tf.constant(0,dtype='float32', shape=[samplePerUpdate])

            aff = [s_x, r_1, loc_sample[:,0], r_2, s_y, loc_sample[:,1]]
            # aff = tf.stop_gradient(aff)
            # print(aff)
            finAff = tf.pack(aff)
            finAff = tf.transpose(finAff, [1, 0])
            # print(finAff.get_shape())

            # %% We'll create a spatial transformer module to identify discriminative
            # %% patches
            out_size = (imgH, imgW)
            stn_out = transformer(images, finAff, out_size)
            stn_out = tf.reshape(stn_out, [samplePerUpdate, imgH, imgW, 1])
            mean, var = tf.nn.moments(stn_out, axes=[1,2], keep_dims=True)
            # print(mean.get_shape())
            stn_out = tf.nn.batch_normalization(stn_out, mean=mean, variance=var, offset=None, scale=None, variance_epsilon=1e-6)
            # stn_out = tf.stop_gradient(stn_out)
    # return readNet(stn_out, phase_train, seqLen), stn_out, loc_sample, loc_mean, b_fc_loc2
    return readNet(stn_out, targetY, seqLen) + (stn_out, loc_sample, loc_mean, b_fc_loc2)

def readNet(stn_out, targetY, seqLen):
    with tf.variable_scope('readPart') as scope:
        with tf.variable_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([6, 5, channels, 32], stddev=5e-2), name='weights')
            ##Weight Decay?
            # weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.002, name='weight_loss')
            # tf.add_to_collection('losses', weight_decay)
            conv = tf.nn.conv2d(stn_out, kernel, [1, 4, 3, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.1, shape=[32]), name='biases')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1_bn = batch_norm(pre_activation, decay=0.999, is_training=False, scope="BN1")
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
            conv2_bn = batch_norm(pre_activation, decay=0.999, is_training=False, scope="BN2")
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
            conv3_bn = batch_norm(pre_activation, decay=0.999, is_training=False, scope="BN3")
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
            droppedFW = rnn_cell.DropoutWrapper(forwardH1, output_keep_prob=1.0)
            backwardH1 = rnn_cell.LSTMCell(nHiddenLSTM1, use_peepholes=True, state_is_tuple=True)
            droppedBW = rnn_cell.DropoutWrapper(backwardH1, output_keep_prob=1.0)
            outputs, _, _ = bidirectional_rnn(droppedFW, droppedBW, rnnIn, dtype=tf.float32)
            fbH1rs = [tf.reshape(t, [samplePerUpdate, 2, nHiddenLSTM1]) for t in outputs]
            # outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]
            outH1 = [tf.reduce_sum(t, reduction_indices=1) for t in fbH1rs]
        with tf.variable_scope('LOGIT') as scope:
            weightsClasses = tf.Variable(tf.truncated_normal([nHiddenLSTM1, nClasses],
                                                             stddev=np.sqrt(2.0 / nHiddenLSTM1)))
            biasesClasses = tf.Variable(tf.zeros([nClasses]))
            logitsFin = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

            logits3d = tf.pack(logitsFin)
            pred = tf.to_int32(ctc.ctc_greedy_decoder(logits3d, seqLenAfterConv)[0][0])
            edist = tf.edit_distance(pred, targetY, normalize=False)
        return pred, edist


# to use for maximum likelihood with input location
def gaussian_pdf(mean, sample):
    Z = 1.0 / (loc_std * tf.sqrt(2.0 * np.pi))
    a = -tf.square(sample - mean) / (2.0 * tf.square(loc_std))
    return Z * tf.exp(a)

def loglikelihood(mean_arr, sampled_arr):
    p_loc = gaussian_pdf(mean_arr, sampled_arr)
    # print(p_loc.get_shape())
    logll = tf.log(tf.reduce_sum(p_loc, 1)) # [batch_sz]
    # print(logll.get_shape())
    # logll = tf.transpose(logll)  # [batch_sz, timesteps]
    return logll

def loss(rewards, loc_s, loc_m):
    averaged = ema.average(singleRew)
    logll = loglikelihood(loc_m, loc_s)

    # advs = rewards - tf.stop_gradient(averaged)
    advs = rewards - averaged
    advs = tf.stop_gradient(advs)
    rewardedPDF = logll * advs
    logllratio = tf.reduce_sum(rewardedPDF)
    return -logllratio


print('Defining graph')
graph = tf.Graph()
with graph.as_default():
    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=(samplePerUpdate, imgInH, imgInW, channels))
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = imgW*tf.ones(shape=(samplePerUpdate))
    is_training = tf.placeholder_with_default(tf.constant(False), [])
    pr, eD, s_o, loc_s, loc_m, b_l2 = inference(inputX, targetY, seqLengths, is_training)
    eD = tf.reshape(eD, (samplePerUpdate,))
    tgtLens = tf.to_float(tf.size(targetY.values))
    err = tf.reduce_sum(eD) / tgtLens
    # eD = tf.stop_gradient(eD)
    rewards = tf.clip_by_value(1.0 - eD, 0., 1.)

    singleRew = tf.reduce_mean(rewards)
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    maintain_averages_op = ema.apply([singleRew])

    loss = loss(rewards, loc_s, loc_m)

    dict1 = get_op('readPart')
    saver1 = tf.train.Saver(dict1)
    # #Optimize ONLY new vars.
    toOpt = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='findPart')
    print('To Train')
    for v in toOpt:
        print(v.name)
    # optimizer = tf.train.AdamOptimizer().minimize(loss, var_list=toOpt)
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, toOpt), grad_norm)
    optimizer = tf.train.AdamOptimizer()
    opt_op = optimizer.apply_gradients(zip(grads, toOpt))


    # optimizer = tf.train.MomentumOptimizer(learning_rate=learningRate, momentum=momentum).minimize(loss, var_list=toOpt)


    with tf.control_dependencies([opt_op]):
        training_op = tf.group(maintain_averages_op)

    # all_varsS = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    # all_vars = []
    # print('ALL VARS')
    # for v in all_varsS:
    #     print(v.name)
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
        if(epoch == 2):
            trainList.extend(trainList1)
        if (epoch == 5):
            trainList.extend(trainList2)
        if (epoch == 20):
            trainList.extend(trainList3)
        if (epoch == 40):
            trainList.extend(trainList4)
        if (epoch == 60):
            trainList.extend(trainList5)

        workList = trainList[:]
        shuffle(workList)
        workList = workList[0:numT]
        print('Epoch', epoch + 1, '...')
        rewT = 0
        lossT = 0
        timeTS = time.time()
        for bStep in range(stepsPerEpocheTrain):
            bList, workList = workList[:batchSize], workList[batchSize:]
            bList2 = []
            for idx in range(len(bList)):
                for hIdx in range(sampleNum):
                    bList2.append(bList[idx])
            batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList2, cm, imgInW,
                                                                                                             mvn=True)
            # labels = np.tile(labels, [config.M])
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, is_training: True}
            _, lossO, rewO = session.run([training_op, loss, singleRew], feed_dict=feedDict)
            # lossO, rewO, loc_sO = session.run([loss, singleRew, loc_s], feed_dict=feedDict)
            rewT += rewO
            lossT += lossO
        print('Train: CTC-loss ', lossT)
        rewTavg = rewT / stepsPerEpocheTrain
        print('Train: Rew ', rewTavg)
        print('Train time ', time.time() - timeTS)
        #Evaluation TrainList
        workList = trainList[:]
        shuffle(workList)
        workList = workList[0:numT]
        rewValT = 0
        lossVT = 0
        errVT = 0
        timeVS = time.time()
        for bStep in range(stepsPerEpocheTrain/sampleNum):
            bList, workList = workList[:samplePerUpdate], workList[samplePerUpdate:]
            batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
                                                                                                             imgInW,
                                                                                                             mvn=True)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, is_training: False}
            lossO, rewO, errO = session.run([loss, singleRew, err], feed_dict=feedDict)
            lossVT += lossO
            rewValT += rewO
            errVT += errO
        print('ValT: CTC-loss ', lossVT)
        rewValAvg = rewValT / stepsPerEpocheTrain*sampleNum
        print('ValT: Rew ', rewValAvg)
        errValT = errVT / stepsPerEpocheTrain*sampleNum
        print('ValT: CER ', errValT)
        print('Val time ', time.time() - timeVS)
        workList = valList[:]
        rewVal = 0
        lossV = 0
        errV = 0
        timeVS = time.time()
        for bStep in range(stepsPerEpocheVal):
            bList, workList = workList[:samplePerUpdate], workList[samplePerUpdate:]
            batchInputs, batchSeqLengths, batchTargetIdxs, batchTargetVals, batchTargetShape = get_list_vals(bList, cm,
                                                                                                             imgInW,
                                                                                                             mvn=True)
            feedDict = {inputX: batchInputs, targetIxs: batchTargetIdxs, targetVals: batchTargetVals,
                        targetShape: batchTargetShape, is_training: False}
            lossO, rewO, errO = session.run([loss, singleRew, err], feed_dict=feedDict)
            lossV += lossO
            rewVal += rewO
            errV += errO
        print('Val: CTC-loss ', lossV)
        rewValAvg = rewVal / stepsPerEpocheVal
        print('Val: Rew ', rewValAvg)
        errVal = errV / stepsPerEpocheVal
        print('Val: CER ', errVal)
        print('Val time ', time.time() - timeVS)
        # Write a checkpoint.
        checkpoint_file = os.path.join('./private/models/lp_stn2/', 'checkpoint')
        saver.save(session, checkpoint_file, global_step=epoch)
