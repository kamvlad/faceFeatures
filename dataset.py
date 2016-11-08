from random import randint

import pandas as pd
import numpy as np
import geometry

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

featureName = { LEFT_EYE_CENTER : 'left_eye_center',
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

class Face:
    def __init__(self, id, sample):
        self.id = id
        imageData = [int(x) for x in sample['Image'].split(' ')]
        self.image = np.array(imageData)
        self.image = np.reshape(self.image, IMAGE_SIZE)
        self.features = {}

        for key, value in featureName.items():
            x = sample[value + '_x']
            y = sample[value + '_y']
            self.features[key] = (x, y)
    def getFeatureImage(self, featureId, rectSize):
        pos = self.features[featureId]
        rect = geometry.rectangleByCenter(pos[0], pos[1], rectSize[0], rectSize[1]).coordsTupleInt()
        return rect.sliceImage(self.image)
    def getFeaturesList(self):
        rslt = []
        for key, value in self.features.items():
            rslt += [value]
        return rslt
    def distSquare(self, featureId, position):
        p = self.features[featureId]
        x = p[0] - position[0]
        y = p[1] - position[1]
        return x * x + y * y

class FacesDB:
    def __init__(self, filename = 'training.csv'):
        self.dataset = pd.read_csv(filename)
        self.faces = {}
    def rows(self):
        return len(self.dataset)
    def getFace(self, imageId):
        if not imageId in self.faces:
            self.faces[imageId] = Face(imageId, self.dataset.iloc[imageId])
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
        return rectCandidate.sliceImage(face.image)

#TODO Implement it
class TestResults:
    def __init__(self, testFilname='test.csv', lookupTableFilename = 'IdLookupTable.csv'):
        self.lockupTable = pd.read_csv(lookupTableFilename)
        self.table = [0.0] * len(self.lockupTable)


    def setValue(self, imageId, feature, value):
        rowId = self.lockupTable[
            (self.lockupTable['ImageId'] == imageId) & (self.lockupTable['FeatureName'] == name)]
        if len(rowId) != 0:
            self.table[rowId.iloc[0]['RowId'] - 1] = value
            return True
        else:
            return False

    def write(self, filename='test_results.csv'):
        fd = open(filename, 'w')
        fd.write('RowId,Location\n')
        for i in range(len(self.table)):
            fd.write("%d,%f\n" % (i + 1, self.table[i]))
        fd.close()