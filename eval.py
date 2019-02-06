from skimage.io import imread, imshow, imsave
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import os, sys, glob
from collections import namedtuple
import fnmatch

def getCsFileInfo(fileName):
    CsFile = namedtuple('csFile', ['city', 'sequenceNb', 'frameNb', 'type', 'type2', 'ext'])
    baseName = os.path.basename(fileName)
    parts = baseName.split('_')
    parts = parts[:-1] + parts[-1].split('.')
    if len(parts) == 5:
        csFile = CsFile( *parts[:-1] , type2="" , ext=parts[-1] )
    elif len(parts) == 6:
        csFile = CsFile( *parts )
    return csFile

def getPrediction(groundTruthFile ):
    predictionPath = os.path.join(os.environ['CITYSCAPES_DATASET'], "results")
    walk = []
    for root, dirnames, filenames in os.walk(predictionPath):
        walk.append((root,filenames))
    predictionWalk = walk

    csFile = getCsFileInfo(groundTruthFile)
    filePattern = "{}_{}_{}*.png".format( csFile.city , csFile.sequenceNb , csFile.frameNb )
    predictionFile = None

    for root, filenames in predictionWalk:
        for filename in fnmatch.filter(filenames, filePattern):
            if not predictionFile:
                predictionFile = os.path.join(root, filename)

    return predictionFile


classes = ['road'   ,
'sidewalk',
'building' ,
'wall'      ,
'fence'      ,
'pole'        ,
'traffic light',
'traffic sign'  ,
'vegetation'    ,
'terrain'       ,
'sky'           ,
'person'        ,
'rider'         ,
'car'           ,
'truck'         ,
'bus'           ,
'train'         ,
'motorcycle'    ,
'bicycle'    ]

def eval_preds(preds, groundTruth):
    eps = 1e-8
    """path = os.environ['CITYSCAPES_DATASET']
    predictionImgList = []
    groundTruthImgList = []
    groundTruthSearch = os.path.join(path, "gtFine", "val", "*", "*_gtFine_labelTrainIds.png")
    groundTruthImgList = glob.glob(groundTruthSearch)"""
    class_ious = np.zeros(20)
    conf_mtrx=np.zeros((20,20))
    """for gt in groundTruthImgList:
        predictionImgList.append(getPrediction(gt))"""
    iou_score = 0
    for i in range(preds.shape[0]):
        sys.stdout.write('\r')
        sys.stdout.write("processed %d images" % i)
        sys.stdout.flush()
        pred = preds[i]
        gt = groundTruth[i] #(groundTruthImgList[i])
        for j in range(20):
            pred1 = (pred == j)
            gt1 = (gt == j)
            intersect = np.logical_and(pred1, gt1).sum()
            union = np.logical_or(pred1, gt1).sum()
            if union != 0:
                class_ious[j]+=(intersect)/(union)
    class_ious/=preds.shape[0]
    print()
    for i in range(len(classes)):
        print(classes[i],':',class_ious[i])
    return np.mean(class_ious)



