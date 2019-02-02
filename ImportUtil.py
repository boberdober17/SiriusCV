from __future__ import print_function, absolute_import, division
import os, glob, sys
from skimage.io import imread, imshow, imsave
import cityscapescripts
def getData(num_tests, start):
    if 'CITYSCAPES_DATASET' in os.environ:
        cityscapesPath = os.environ['CITYSCAPES_DATASET']
    else:
        cityscapesPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    searchAnnotated = os.path.join(cityscapesPath, "gtFine", "train", "*", "*_gt*_labelTrain*")
    searchRaw = os.path.join(cityscapesPath, "img8bit", "train", "*", "*.png")
    filesAnnotated =glob.glob(searchAnnotated)
    filesRaw=glob.glob(searchRaw)
    filesAnnotated.sort()
    filesRaw.sort()
    """
    edit files so that filesAnnotated and filesRaw are the same shape and correspond to each other
    """
    return (filesAnnotated[start:start+num_tests], filesRaw[start:start+num_tests])

def importBatch(num_tests, start):   #load batch of data from train dataset
    y_files, X_files = getData(num_tests,start)
    X_input = []
    y_input = []
    for i in range(X_files.shape[0]):
        y_file = y_files[i]
        X_file = X_files[i]
        X_img = imread(X_file)
        y_img = imread(y_file)
        X_input.append(X_img)
        y_input.append(y_img)
    return X_input, y_input
def initTrain():
    import cityscapesscripts.preparation.createTrainIdLabelImgs
    import cityscapesscripts.preparation.createTrainIdInstanceImgs




