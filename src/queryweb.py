from .VLADlib.VLAD import *
from .VLADlib.Descriptors import *
import itertools
import glob
import cv2
import os
import numpy as np


def queryweb(queryName, x, y, w, h, crop = True):
    # path
    pathCurr = os.path.abspath(os.path.dirname(__file__))
    dir_images = os.path.join(pathCurr,"dataset","img")

    # arg query
    k = int(10) # top 10 
    descriptorName = "ORB"
    pathVD = os.path.join(pathCurr, "visualDictionary", "visualDictionary64ORB.pickle")
    treeIndex = os.path.join(pathCurr, "ballTreeIndexes", "index_ORB_W64.pickle")
    
    #load the index
    with open(treeIndex, 'rb') as f:
        indexStructure=pickle.load(f)

    #load the visual dictionary
    with open(pathVD, 'rb') as f:
        visualDictionary=pickle.load(f)     

    imageID=indexStructure[0]
    tree = indexStructure[1]
    pathImageData = indexStructure[2]

    rank_file = 'rank.txt'

    path = os.path.join(dir_images, queryName)
    dist, ind = query(path, k, descriptorName, visualDictionary, tree, x, y, w, h, crop)
    ind=list(itertools.chain.from_iterable(ind))
    rank_names = []
    for j in ind:
        rank_names.append(imageID[j].split("\\",1)[1])

    # write rank names to txt
    f = open(rank_file, 'w')
    for i, name in enumerate(rank_names):
        f.writelines(name + ' ' + str(dist[0][i])+ '\n')
    f.close()

