# -*- coding: utf-8 -*-
from __future__ import print_function

# from idlelib.PyParse import ch

import matplotlib.pyplot as plt
import numpy as np
from __builtin__ import unicode
from numpy.core.numerictypes import val
from pip import index
from scipy import misc
import os
import codecs
import string


def get_cm_lp():
    cm = CharacterMapper()
    for x in range(10):
        cm.add(str(x))
    for Z in string.ascii_uppercase:
        cm.add(Z)
    cm.add(u'Ä')
    cm.add(u'Ö')
    cm.add(u'Ü')
    idx = cm.add(u'-')
    cm.add(u'_', idx)
    cm.add(u' ', idx)
    cm.add(u'.', idx)
    return cm

def get_cm_iam():
    cm = CharacterMapper()
    # for x in range(10):
    #     cm.add(str(x))
    for Z in string.printable:
        cm.add(Z)
    # for Z in string.ascii_lowercase:
    #     cm.add(Z)
    # cm.add(u'Ä')
    # cm.add(u'Ö')
    # cm.add(u'Ü')
    # idx = cm.add(u'-')
    # cm.add(u'_', idx)
    # cm.add(u' ', idx)
    # cm.add(u' ')
    # cm.add(u'.')
    # cm.add(u',')
    # cm.add(u'?')
    # cm.add(u'!')
    # cm.add(u'-')
    # cm.add(u"'")
    # cm.add(u'"')
    # cm.add(u'(')
    # cm.add(u')')
    # cm.add(u':')
    # cm.add(u';')
    # cm.add(u'&')
    # cm.add(u'/')
    return cm


class CharacterMapper:
    def __init__(self):
        self.dictFwd = {}
        self.dictBwd = {}
        self.loaded = False

    def get_channel(self, character):
        return self.dictFwd[character]

    def get_channels(self, characters):
        res = []
        if type(characters) == str:
            characters = characters.decode('utf-8')
        for v in range(len(characters)):
            print(characters[v])
            res.append(self.get_channel(characters[v]))
        return res

    def get_value(self, channel):
        return self.dictBwd[channel]

    def get_values(self, channels):
        channels = np.reshape(channels,newshape=[-1])
        res = []
        for v in range(len(channels)):
            res.append(self.get_value(channels[v]))
        return res

    def size(self):
        return len(self.dictBwd)

    def add(self, unicode, channel=-1):
        """

        :param unicode:
        :type channel: if not given, use lenght of dict
        """
        if channel == -1:
            channel = len(self.dictFwd)
            # print("len = " + str(index))
        self.dictFwd[unicode] = channel
        if not self.dictBwd.has_key(channel):
            self.dictBwd[channel] = unicode
        # print("pos = " + str(index))
        return channel

    def get_mapping(file):
        with codecs.open(file, 'r', encoding='utf-8') as file:
            raw = file.readlines()
            print(raw)
            return raw

    def load_mapping(self, file):
        if self.loaded:
            raise RuntimeError("map already loaded")
        with codecs.open(file, 'r', encoding='utf-8') as file:
            raw = file.readlines()
            for line in raw:
                if line[-1] == '\n':
                    line = line[:-1]
                print(line)
                split = line.rsplit('=', 1)
                key = split[0]
                index = int(split[1]) - 1
                # specific values which are escaped by '\': delete '\'
                if key[0] == '\\':
                    key = key[1:]
                print("'" + key + "' ==> " + str(index))
                self.add(key, index)
        self.loaded = True

    def print(self):
        # for key in self.dictBwd.keys():
        # print(key)
        # print(str(key)+" => "+(self.dictBwd.get(key)))
        print(self.dictFwd)
        print(self.dictBwd)

# if __name__ == '__main__':
#     os.chdir("..")
#     # print("bla")
#     # cm = CharacterMapper()
#     # cm.load_mapping('private/data/barlach/cm.txt')
#     # cm.print()
#     # print("value = " + str(cm.get_channel('-')))
#     # print("value of " + (u'\u2014') + " = " + str(cm.get_channel(u'\u2014')))
#     # print("value of " + (u'\xe4') + " = " + str(cm.get_channel(u'\xe4')))
#     # print("#############################################")
#     cm = get_cm_lp()
#     cm.print()
#     print("cm = " + str(cm.dictFwd))
#     print(cm.get_channels(u'012-Ö4'))
