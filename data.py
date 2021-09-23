from keras.backend import expand_dims
from keras.datasets.fashion_mnist import load_data
from numpy.random import randint
from numpy.random import randn
from numpy import zeros
from numpy import ones
from tensorflow.keras.models import Model

class Data():
    def __init__(self) -> None:
        self.dataset = self.loadDataset()

    def loadDataset(self):
        (trainX, y), (_, _) = load_data()
        X = expand_dims(trainX, axis=-1)
        X = X.numpy().astype('float32')
        X = (X - 127.5) / 127.5 # scale from 0,255 to -1,1
        return [X, y]

    def generateRealTrainingSamples(self, samples):
        images, labels = self.dataset
        ix = randint(0, images.shape[0], samples)
        X, labels = images[ix], labels[ix]
        y = ones((samples, 1))
        return [X, labels], y

    def generateFakeTrainingSamples(self, generator:Model, latentDim, samples):
        x, labels = self.generateLatentPoints(latentDim, samples)
        X = generator.predict([x, labels])
        y = zeros((samples, 1))
        return [X, labels], y

    def generateFakeTrainingGanSamples(self, latentDim, samples):
        X, labels = self.generateLatentPoints(latentDim, samples)
        y = ones((samples, 1))
        return [X, labels], y

    def generateLatentPoints(self, latentDim, samples, classes=10):
        x = randn(latentDim * samples)
        x = x.reshape((samples, latentDim))
        labels = randint(0, classes, samples)
        return [x, labels]