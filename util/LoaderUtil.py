from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
import STR2CTC
import os
import codecs


def read_image_list(pathToList):
    """Reads a .txt file containing paths to the images
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(pathToList, 'r')
    filenames = []
    for line in f:
        if line[-1] == '\n':
            filenames.append(line[:-1])
        else:
            filenames.append(line)
    f.close()
    return filenames


def get_batch_labels(bList, cm):
    u_labels = []
    for path in bList:
        labelFile = path[:] + ".txt"
        tmp = codecs.open(labelFile, 'r', encoding='utf-8')
        u_str = tmp.readline()
        u_labels.append(u_str)
        # print(str)
        if tmp is not None:
            tmp.close()
    idx, val, shape = STR2CTC.target_string_list_to_ctc_tensor_repr(u_labels, cm)
    return idx, val, shape


def get_batch_imgs(bList, imgW, mvn):
    imgs = []
    seqL = []

    # print("\n")
    for path in bList:
        # print(path)
        aImg = misc.imread(path)
        width = aImg.shape[1]
        hei = aImg.shape[0]
        # aSeqL = min(width, imgW)
        # aSeqL = max(aSeqL, imgW / 2)
        aSeqL = width
        seqL.append(aSeqL)
        # aImg = aImg.astype('float32')
        aImg = aImg / 255.0
        if mvn:
            std = np.std(aImg)
            mean = np.mean(aImg)
            tImg = (aImg - mean) / std
            aImg = tImg
        if width < imgW:
            padW = imgW - width
            npad = ((0, 0), (0, padW))
            tImg = np.pad(aImg, npad, mode='constant', constant_values=0)
            aImg = tImg
        # if width > imgW:
        #     tImg = aImg[:, :imgW]
        #     aImg = tImg
        # plt.imshow(aImg, cmap=plt.cm.gray)
        # plt.show()
        imgs.append(aImg)

    bSize = len(bList)
    imgBatched = np.zeros((bSize, hei, imgW, 1), dtype='float32')
    # batch the image list
    for idx, img in enumerate(imgs):
        imgBatched[idx, :, :, 0] = img

    return imgBatched, seqL


def get_list_vals(bList, cm, imgW, mvn=False):
    tgtIdx, tgtVal, tgtShape = get_batch_labels(bList, cm)
    inpBatch, inpSeqL = get_batch_imgs(bList, imgW, mvn)
    return inpBatch, inpSeqL, tgtIdx, tgtVal, tgtShape


def clean_list(list, imgW, cm, subsampling=-1):
    res = []
    # Count the skipped Images (width too big)
    countW = 0
    # Count the skipped Images (char not in charMap)
    countC = 0
    # Count the skipped Images (subsampling too much)
    countS = 0
    for path in list:
        aImg = misc.imread(path)
        width = aImg.shape[1]

        # Skip image if image width is bigger than imgW
        if width > imgW:
            countW += 1
            continue

        # Skip image if a character is not in charMap
        skip = False
        labelFile = path[:] + ".txt"
        tmp = codecs.open(labelFile, 'r', encoding='utf-8')
        u_str = tmp.readline()
        if subsampling > 0:
            if subsampling * len(u_str) > width:
                countS += 1
                continue
        if tmp is not None:
            tmp.close()
        count = 0
        lastCh = -1
        for c in u_str:
            try:
                ch = cm.get_channel(c)
                if lastCh == ch:
                    count += 1
                lastCh = ch
            except KeyError:
                # print('Character \'{}\' not in charMap, skipping Image...'.format(c))
                skip = True
                countC += 1
                break
        if not skip:
            if subsampling * (count + len(u_str)) > width:
                countS += 1
                continue
            res.append(path)
    print("Skipped {} out of {} images...".format(countC + countW + countS, len(list)))
    print("...{} too big images, {} images where subsampling is too much and additionally {} images with unknown characters.".format(countW, countS, countC))
    return res


if __name__ == '__main__':
    os.chdir("..")
    list = read_image_list('./resources/lp_only_train.lst')
    imgBatches, seqL = get_list_vals(list, STR2CTC.get_charmap_lp(), 100)
    # print(seqL)
    print(imgBatches.shape)
    print(imgBatches.dtype)
    plt.imshow(imgBatches[129], cmap=plt.cm.gray)
    plt.show()
