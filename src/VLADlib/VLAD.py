# implementation of VLAD ballTree algorithms for CBIR
# Jorge Guevara - jorged@br.ibm.com
# Mikel Bober-Irizar - mikel@mxbi.net

import numpy as np
import itertools
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import BallTree
import pickle
import glob
import cv2
from .Descriptors import *
from tqdm import tqdm
import multiprocessing


# inputs
# getDescriptors for whole dataset
# Path = path to the image dataset
# functionHandleDescriptor={describeSURF, describeSIFT, describeORB}

def getSingleImageDescriptor(args):
    imagePath, functionHandleDescriptor = args
    im = cv2.imread(imagePath)
    kp, des = functionHandleDescriptor(im)
    return des

def getDescriptors(path, functionHandleDescriptor, threads):
    print('Searching for images...')
    files = glob.glob(path + "/*.jpg")
    print('Found {} images...'.format(len(files)))

    print('Running with {} threads'.format(threads))

    if threads == 1:
        descriptors = []
        for imagePath in tqdm(files, desc="Calculating descriptors"):
            im = cv2.imread(imagePath)
            kp, des = functionHandleDescriptor(im)
            if des is not None:
                descriptors.append(des)

        #flatten list
        descriptors = list(itertools.chain.from_iterable(descriptors))
    else:
        pool = multiprocessing.Pool(threads)

        data = [(f, functionHandleDescriptor) for f in files] # Generate payload to send to threads
        descriptors = []
        for descs in tqdm(pool.imap_unordered(getSingleImageDescriptor, data), desc="[{} CPUs] Calculating descriptors".format(threads), total=len(data)):
            if descs is not None:
                descriptors.extend(descs)
        pool.close()

    #list to array
    descriptors = np.asarray(descriptors)

    return descriptors


# input
# training = a set of descriptors
def kMeansDictionary(training, k):

    #K-means algorithm
    #est = KMeans(n_clusters=k, init='k-means++', tol=0.0001, verbose=1).fit(training)
    est = MiniBatchKMeans(n_clusters=k, init='k-means++', max_iter=100, batch_size=100, verbose=1,tol=0.0).fit(training)
    #centers = est.cluster_centers_
    #labels = est.labels_
    #est.predict(X)
    return est
    #clf2 = pickle.loads(s)

# compute vlad descriptors for te whole dataset
# input: path = path of the dataset
#        functionHandleDescriptor={describeSURF, describeSIFT, describeORB}
#        visualDictionary = a visual dictionary from k-means algorithm


def getVLADDescriptors(path,functionHandleDescriptor,visualDictionary):
    descriptors=list()
    idImage =list()
    for imagePath in glob.glob(path+"/*.jpg"):
        print(imagePath)
        im=cv2.imread(imagePath)
        kp,des = functionHandleDescriptor(im)
        if not (des is None):
            v=VLAD(des,visualDictionary)
            descriptors.append(v)
            idImage.append(imagePath)

    #list to array
    descriptors = np.asarray(descriptors)
    return descriptors, idImage


# fget a VLAD descriptor for a particular image
# input: X = descriptors of an image (M x D matrix)
# visualDictionary = precomputed visual dictionary

# compute vlad descriptors per PDF for te whole dataset, f
# input: path = dataset path
#        functionHandleDescriptor={describeSURF, describeSIFT, describeORB}
#        visualDictionary = a visual dictionary from k-means algorithm


def getVLADDescriptorsPerPDF(path,functionHandleDescriptor,visualDictionary):
    descriptors=list()
    idPDF =list()
    desPDF= list()

    #####
    #sorting the data
    data=list()
    for e in glob.glob(path+"/*.jpg"):
        #print("e: {}".format(e))
        s=e.split('/')
        #print("s: {}".format(s))
        s=s[1].split('-')
        #print("s: {}".format(s))
        s=s[0].split('.')
        #print("s: {}".format(s))
        s=int(s[0]+s[1])
        #print("s: {}".format(s))

        data.append([s,e])

    data=sorted(data, key=lambda atr: atr[0])
    #####

    #sFirst=glob.glob(path+"/*.jpg")[0].split('-')[0]
    sFirst=data[0][0]
    docCont=0
    docProcessed=0
    #for imagePath in glob.glob(path+"/*.jpg"):
    for s, imagePath in data:
        #print(imagePath)
        #s=imagePath.split('-')[0]
        #print("s : {}".format(s))
        #print("sFirst : {}".format(sFirst))

        #accumulate all pdf's image descriptors in a list
        if (s==sFirst):

            im=cv2.imread(imagePath)
            kp,des = functionHandleDescriptor(im)
            if des!=None:
                desPDF.append(des)

        else:
            docCont=docCont+1
            #compute VLAD for all the descriptors whithin a PDF
            #------------------
            if len(desPDF)!=0:
                docProcessed=docProcessed+1
                #print("len desPDF: {}".format(len(desPDF)))
                #flatten list
                desPDF = list(itertools.chain.from_iterable(desPDF))
                #list to array
                desPDF = np.asarray(desPDF)
                #VLAD per PDF
                v=VLAD(desPDF,visualDictionary)
                descriptors.append(v)
                idPDF.append(sFirst)
            #------------------
            #update vars
            desPDF= list()
            sFirst=s
            im=cv2.imread(imagePath)
            kp,des = functionHandleDescriptor(im)
            if des!=None:
                desPDF.append(des)

    #Last element
    docCont=docCont+1
    if len(desPDF)!=0:
        docProcessed=docProcessed+1
        desPDF = list(itertools.chain.from_iterable(desPDF))
        desPDF = np.asarray(desPDF)
        v=VLAD(desPDF,visualDictionary)
        descriptors.append(v)
        idPDF.append(sFirst)

    #list to array
    descriptors = np.asarray(descriptors)
    print("descriptors: {}".format(descriptors))
    print("idPDF: {}".format(idPDF))
    print("len descriptors : {}".format(descriptors.shape))
    print("len idpDF: {}".format(len(idPDF)))
    print("total number of PDF's: {}".format(docCont))
    print("processed number of PDF's: {}".format(docProcessed))

    return descriptors, idPDF


# fget a VLAD descriptor for a particular image
# input: X = descriptors of an image (M x D matrix)
# visualDictionary = precomputed visual dictionary

def VLAD(X,visualDictionary):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters

    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)


    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V



#Implementation of a improved version of VLAD
#reference: Revisiting the VLAD image representation
def improvedVLAD(X,visualDictionary):

    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters

    m,d = X.shape
    V=np.zeros([k,d])
    #computing the differences

    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            # add the diferences
            V[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)


    V = V.flatten()
    # power normalization, also called square-rooting normalization
    V = np.sign(V)*np.sqrt(np.abs(V))

    # L2 normalization

    V = V/np.sqrt(np.dot(V,V))
    return V

def indexBallTree(X,leafSize):
    tree = BallTree(X, leaf_size=leafSize)
    return tree

#typeDescriptors =SURF, SIFT, OEB
#k = number of images to be retrieved
def query(image, k, descriptorName, visualDictionary,tree, x, y, w, h, cropped = True):
    #read image
    im=cv2.imread(image)
    if cropped:
        im = im[y:y+h, x:x+w]
    #compute descriptors
    dict={"SURF":describeSURF,"SIFT":describeSIFT,"ORB":describeORB}
    funDescriptor=dict[descriptorName]
    kp, descriptor=funDescriptor(im)
    
    if descriptor is None:
        return [], []
    #compute VLAD
    v=VLAD(descriptor,visualDictionary)

    #find the k most relevant images
    dist, ind = tree.query(v.reshape(1, -1), k)

    return dist, ind
