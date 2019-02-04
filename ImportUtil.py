from __future__ import print_function, absolute_import, division
import os, glob, sys
from skimage.io import imread, imshow, imsave
import numpy as np
import cityscapesscripts
def getData(num_tests, start, type):
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    searchAnnotated = os.path.join(cityscapesPath, "gtFine", type, "*", "*_gt*_labelTrain*")
    searchRaw = os.path.join(cityscapesPath, "leftImg8bit", type, "*", "*.png")

    #if not searchAnnotated:
    #    printError("Did not find any annotated files.")
    filesAnnotated =glob.glob(searchAnnotated)

    filesRaw=glob.glob(searchRaw)
    filesAnnotated.sort()
    filesRaw.sort()

    return filesAnnotated[start:start+num_tests], filesRaw[start:start+num_tests]

def importBatch(num_tests, start, verbose, type="train"):   #load batch of data from train dataset

    y_files, X_files = getData(num_tests,start, type)
    X_input = []
    y_input = []
    if type=='val':
        filenames = []
    z = 0
    for i in range(len(X_files)):

        z+=1
        if verbose:
            if z % 100 == 0:
                print('loaded files input - ', z)

        X_file = X_files[i]
        filenames.append(X_file[:-16])
        X_img = imread(X_file)
        X_input.append(X_img)
    z = 0
    for i in range(len(X_files)):
        z += 1
        if verbose:
            if z % 100 == 0:
               print('loaded files output - ', z)

        y_file = y_files[i]
        y_img = imread(y_file)
        y_input.append(y_img)


    X = np.array(X_input)
    y = np.array(y_input)
    if (type=='val')
        return X,y, filenames
    return X, y
def initTrain():
    import cityscapesscripts.preparation.createTrainIdLabelImgs
    import cityscapesscripts.preparation.createTrainIdInstanceImgs




