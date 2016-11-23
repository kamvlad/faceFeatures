from random import seed, shuffle

import dataset
import numpy as np
from numpy.core.numeric import empty

def scale(x, maxX):
    return 2.0 * x / float(maxX) - 1.0

def main():
    writeNoFullFeatured = False
    shuffleSet = True
    sSeed = 1223455

    seed(sSeed)

    trainingDB = dataset.TrainingsetDB()
    imageSize = dataset.IMAGE_SIZE
    trainingData = np.zeros((trainingDB.rows(), imageSize[0] * imageSize[1]), dtype=np.float32)
    trainingY = empty((trainingDB.rows(), 15 * 2), dtype=np.float32)
    trainingY.fill(np.nan)

    rowId = 0
    for face in trainingDB.facesList():
        if writeNoFullFeatured or face.allFeaturesPresent():
            trainingData[rowId, :] = (face.image.astype(np.float32) / 255.0).reshape(imageSize[0] * imageSize[1])
            for featureId, position in face.features.items():
                trainingY[rowId, featureId * 2] = scale(position[0], imageSize[0])
                trainingY[rowId, featureId * 2 + 1] = scale(position[1], imageSize[1])
            rowId += 1

    print(rowId, ' from ', trainingDB.rows(), ' selected')
    trainingData = trainingData[:rowId,:]
    trainingY =  trainingY[:rowId]

    if shuffleSet:
        idxs = list(range(trainingData.shape[0]))
        idxs = shuffle(idxs)
        trainingData[idxs] = trainingData
        trainingY[idxs] = trainingY

    np.savez('faceFeatures_9216_normalized_labels_as_coord_full_data.npz', data=trainingData, y=trainingY)


if __name__ == '__main__':
    main()