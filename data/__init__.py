import pandas as pd
from random import randint

import numpy as np

from utils import geometry

IMAGE_SIZE = (96, 96)

LEFT_EYE_CENTER = 0
RIGHT_EYE_CENTER = 1
LEFT_EYE_INNER_CORNER = 2
LEFT_EYE_OUTER_CORNER = 3
RIGHT_EYE_INNER_CORNER = 4
RIGHT_EYE_OUTER_CORNER = 5
LEFT_EYEBROW_INNER_END = 6
LEFT_EYEBROW_OUTER_END = 7
RIGHT_EYEBROW_INNER_END = 8
RIGHT_EYEBROW_OUTER_END = 9
NOSE_TIP = 10
MOUTH_LEFT_CORNER = 11
MOUTH_RIGHT_CORNER = 12
MOUTH_CENTER_TOP_LIP = 13
MOUTH_CENTER_BOTTOM_LIP = 14

allFeatures = range(15)
featuresNames = { LEFT_EYE_CENTER : 'left_eye_center',
                RIGHT_EYE_CENTER : 'right_eye_center',
                LEFT_EYE_INNER_CORNER : 'left_eye_inner_corner',
                LEFT_EYE_OUTER_CORNER : 'left_eye_outer_corner',
                RIGHT_EYE_INNER_CORNER : 'right_eye_inner_corner',
                RIGHT_EYE_OUTER_CORNER : 'right_eye_outer_corner',
                LEFT_EYEBROW_INNER_END : 'left_eyebrow_inner_end',
                LEFT_EYEBROW_OUTER_END : 'left_eyebrow_outer_end',
                RIGHT_EYEBROW_INNER_END : 'right_eyebrow_inner_end',
                RIGHT_EYEBROW_OUTER_END : 'right_eyebrow_outer_end',
                NOSE_TIP : 'nose_tip',
                MOUTH_LEFT_CORNER : 'mouth_left_corner',
                MOUTH_RIGHT_CORNER : 'mouth_right_corner',
                MOUTH_CENTER_TOP_LIP : 'mouth_center_top_lip',
                MOUTH_CENTER_BOTTOM_LIP : 'mouth_center_bottom_lip' }

def stringToImage(str):
    imageData = [int(x) for x in str.split(' ')]
    image = np.array(imageData)
    return np.reshape(image, IMAGE_SIZE)

class Face:
    def __init__(self, id, imageData):
        self.id = id
        self.image = stringToImage(imageData)
        self.features = {}
    def setFeaturesFromSample(self, sample):
        for key, value in featuresNames.items():
            x = sample[value + '_x']
            y = sample[value + '_y']
            self.features[key] = (x, y)
    def setFeaturePosition(self, featureId, position):
        self.features[featureId] = position
    def getFeatureImage(self, featureId, rectSize):
        pos = self.features[featureId]
        rect = geometry.rectangleByCenter(pos[0], pos[1], rectSize[0], rectSize[1])
        return rect.sliceImage(self.image)
    def getFeaturesValues(self):
        rslt = []
        for key, value in self.features.items():
            rslt += [value]
        return rslt
    def allFeaturesPresent(self):
        for key, value in self.features.items():
            if np.isnan(value[0]) or np.isnan(value[1]):
                return False
        return True
    def distSquare(self, featureId, position):
        p = self.features[featureId]
        x = p[0] - position[0]
        y = p[1] - position[1]
        return x * x + y * y

class TrainingsetDB:
    def __init__(self, filename = 'data/training.csv'):
        self.dataset = pd.read_csv(filename)
        self.faces = {}
    def rows(self):
        return len(self.dataset)
    def facesList(self):
        r = []
        for row in self.dataset.iterrows():
            r += [self.getFace(row[0] + 1)]
        return r
    def getFace(self, imageId):
        assert (imageId > 0)
        if not imageId in self.faces:
            sample = self.dataset.iloc[imageId - 1]
            self.faces[imageId] = Face(imageId - 1, sample['Image'])
            self.faces[imageId].setFeaturesFromSample(sample)
        return self.faces[imageId]
    def randomSubImage(self, imageId, rectSize, excludeFeatures = [], areaOfOverlapThreshold = 0):
        #TODO test it!
        face = self.getFace(imageId)

        foundOverlap = True
        while foundOverlap:
            gx = randint(0, IMAGE_SIZE[1] - rectSize[0])
            gy = randint(0, IMAGE_SIZE[0] - rectSize[1])
            rectCandidate = geometry.Rectangle(gx, gy, rectSize[0], rectSize[1])

            foundOverlap = False
            for featureId in excludeFeatures:
                x, y = face.features[featureId][0], face.features[featureId][1]
                featureRect = geometry.rectangleByCenter(x, y, rectSize[0], rectSize[1])
                if rectCandidate.ao(featureRect) > areaOfOverlapThreshold:
                    foundOverlap = True
                    break
        return rectCandidate.sliceImage(face.image)

class TestsetDB:
    def __init__(self, testFilename='data/test.csv', lookupTableFilename = 'data/IdLookupTable.csv'):
        self.lockupTable = pd.read_csv(lookupTableFilename)
        imagesData = pd.read_csv(testFilename)
        self.faces = {}
        for row in imagesData.iterrows():
            self.faces[row[1]['ImageId']] = Face(row[1]['ImageId'], row[1]['Image'])
    def rows(self):
        return len(self.faces)
    def getFaces(self):
        return self.faces.values()
    def getFeaturesCount(self, imageId):
        assert (imageId > 0)
        return sum(self.lockupTable['ImageId'] == imageId) / 2
    def getFace(self, imageId):
        assert (imageId > 0)
        return self.faces[imageId]
    def getRowId(self, imageId, featureName):
        assert (imageId > 0)
        rowId = self.lockupTable[
            (self.lockupTable['ImageId'] == imageId) & (self.lockupTable['FeatureName'] == featureName)]
        if len(rowId) != 0:
            return rowId.iloc[0]['RowId'] - 1
        else:
            return None
    def write(self, filename='test_results.csv'):
        table = [0.0] * len(self.lockupTable)
        for imageId, face in self.faces.items():
            for featureId, position in face.features.items():
                rowIdX = self.getRowId(imageId, featuresNames[featureId] + '_x')
                rowIdY = self.getRowId(imageId, featuresNames[featureId] + '_y')
                if rowIdX != None:
                    table[rowIdX] = position[0]
                    table[rowIdY] = position[1]
        fd = open(filename, 'w')
        fd.write('RowId,Location\n')
        for i in range(len(table)):
            fd.write("%d,%f\n" % (i + 1, table[i]))
        fd.close()