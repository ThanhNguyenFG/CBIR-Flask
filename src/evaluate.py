from VLADlib.VLAD import *
from VLADlib.Descriptors import *
import itertools
import glob
import cv2
import os
import numpy as np


def query_images(groundtruth_dir, dataset):

    query_names = []
    fake_query_names = []
    cropped = True
    for f in glob.iglob(os.path.join(groundtruth_dir, '*_query.txt')):
        fake_query_name = os.path.splitext(os.path.basename(f))[0].replace('_query', '')
        fake_query_names.append(fake_query_name)

        query_name, x, y, w, h = open(f).read().strip().split(' ')

        if cropped:
            x, y, w, h = map(float, (x, y, w, h))
            x, y, w, h = map(lambda d: int(round(d)), (x, y, w, h))

        if dataset == 'oxford':
            query_name = query_name.replace('oxc1_', '')
            query_names.append('%s.jpg' % query_name)

    return query_names, fake_query_names, x, y, w, h

if __name__ == '__main__':
    
    # path
    pathCurr = os.path.abspath(os.path.dirname(__file__))
    gt_files = os.path.join(pathCurr,"dataset","gt")
    dir_images = os.path.join(pathCurr,"dataset","img")

    query_names, fake_query_names, x, y, w, h = query_images(gt_files, 'oxford')

    # arg query
    k = int(100)
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

    #print query_names
    aps = []
    rank_file = 'tmp.txt'
  
    for i, queryName in enumerate(query_names):

        path = os.path.join(dir_images, queryName)
        dist,ind = query(path, k, descriptorName, visualDictionary, tree, x, y, w, h, False)
        ind=list(itertools.chain.from_iterable(ind))
        rank_names = []
        for j in ind:
            rank_names.append(imageID[j].split("\\",1)[1])

        # write rank names to txt
        f = open(rank_file, 'w')
        f.writelines([name.split('.jpg')[0] + '\n' for name in rank_names])
        f.close()

        # compute mean average precision
        gt_prefix = os.path.join(gt_files, fake_query_names[i])
        cmd = '.\compute_ap "%s" "%s"' % (gt_prefix, rank_file)
        ap = os.popen(cmd).read()

        #os.remove(rank_file)
        aps.append(float(ap.strip()))

        print("%s, %f" %(queryName, float(ap.strip())))

    print()
    print ("mAP: %f" % np.array(aps).mean())

    