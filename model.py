import os
import time
import datetime

from data import Data
from matplotlib import pyplot
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Reshape, Flatten, Embedding
from tensorflow.keras.layers import ReLU, LeakyReLU, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class ConditionalGAN:
    def __init__(self, data:Data, inputShape, imageDim, labelDim, latentDim, classes, params=None) -> None:
        self.initHyperParameters(params)
        self.initMetricsVars()
        self.evalDirectoryName = 'eval'

        self.data = data
        self.latentDim = latentDim
        self.discriminator = self.createDiscriminator(inputShape, labelDim, classes)
        self.generator = self.createGenerator(latentDim, imageDim, labelDim, classes)
        self.gan = self.createGan()

    def initHyperParameters(self, params):
        self.convFilters = [int(x) for x in params.convFilters.split(',')]
        self.convTransposeFilters = [int(x) for x in params.convTransposeFilters.split(',')]
        self.adamLearningRate = params.adamLearningRate
        self.adamBeta1 = params.adamBeta1
        self.kernelInitStdDev = params.kernelInitStdDev
        self.generatorInputFilters = params.generatorInputFilters
        self.leakyReluAlpha = params.leakyReluAlpha
        self.dropoutRate = params.dropoutRate
        self.convLayerKernelSize = (3,3)
        self.convTransposeLayerKernelSize = (4,4)
        self.generatorOutputLayerKernelSize = (7,7)

    def initMetricsVars(self):
        self.realLossHistory = list()
        self.realAccHistory = list()
        self.fakeLossHistory = list()
        self.fakeAccHistory = list()
        self.lossHistory = list()
        self.metricHistory = list()

    def createDiscriminator(self, inputShape, labelDim, classes, batchNorm=True) -> Model:
        labelInputNodes = inputShape[0] * inputShape[1]
        init = RandomNormal(stddev=self.kernelInitStdDev)

        labelInput = Input(shape=(1,))
        labelEmbedding = Embedding(classes, labelDim)(labelInput)
        labelDense = Dense(labelInputNodes)(labelEmbedding)
        labelShaped = Reshape((inputShape[0], inputShape[1], 1))(labelDense)

        imageInput = Input(shape=inputShape)
        imageLabelConcat = Concatenate()([imageInput, labelShaped])

        convLayer = self.buildConvLayers(batchNorm, init, imageLabelConcat)

        flattenLayer = Flatten()(convLayer)
        outputLayer = Dense(1, activation='sigmoid')(flattenLayer)
        model = Model([imageInput, labelInput], outputLayer)

        opt = Adam(learning_rate=self.adamLearningRate, beta_1=self.adamBeta1)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def createGenerator(self, latentDim, imageDim, labelDim, classes, batchNorm=True) -> Model:
        labelInputNodes = imageDim * imageDim
        init = RandomNormal(stddev=self.kernelInitStdDev)

        labelInput = Input(shape=(1,))
        labelEmbedding = Embedding(classes, labelDim)(labelInput)
        labelDense = Dense(labelInputNodes, kernel_initializer=init)(labelEmbedding)
        labelShaped = Reshape((imageDim, imageDim, 1))(labelDense)

        imageInputNodes = self.generatorInputFilters * imageDim * imageDim
        imageInput = Input(shape=(latentDim,))
        imageDense = Dense(imageInputNodes, kernel_initializer=init)(imageInput)
        imageActv = LeakyReLU(self.leakyReluAlpha)(imageDense)
        imageShaped = Reshape((imageDim, imageDim, self.generatorInputFilters))(imageActv)

        imageLabelConcat = Concatenate()([imageShaped, labelShaped])

        convLayer = self.buildConvTransposeLayers(batchNorm, init, imageLabelConcat)

        outputLayer = Conv2D(1, self.generatorOutputLayerKernelSize, padding='same', activation='tanh', kernel_initializer=init)(convLayer)
        model = Model([imageInput, labelInput], outputLayer)
        return model

    def createGan(self) -> Model:
        for layer in self.discriminator.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False

        genImageInput, genLabelInput = self.generator.input
        genOutput = self.generator.output

        ganOutput = self.discriminator([genOutput, genLabelInput])
        model = Model([genImageInput, genLabelInput], ganOutput)

        opt = Adam(learning_rate=self.adamLearningRate, beta_1=self.adamBeta1)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def buildConvLayer(self, filters, batchNorm, kernelInit, inLayer):
        layer = Conv2D(filters, self.convLayerKernelSize, (2,2), padding='same', kernel_initializer=kernelInit)(inLayer)
        if batchNorm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU(self.leakyReluAlpha)(layer)
        outLayer = Dropout(self.dropoutRate)(layer)
        return outLayer

    def buildConvTransposeLayer(self, filters, batchNorm, kernelInit, inLayer):
        layer = Conv2DTranspose(filters, self.convTransposeLayerKernelSize, (2,2), padding='same', kernel_initializer=kernelInit)(inLayer)
        if batchNorm:
            layer = BatchNormalization()(layer)
        layer = LeakyReLU(self.leakyReluAlpha)(layer)
        outLayer = Dropout(self.dropoutRate)(layer)
        return outLayer

    def buildConvLayers(self, batchNorm, kernelInit, inLayer):
        layer = inLayer
        for f in self.convFilters:
            layer = self.buildConvLayer(f, batchNorm, kernelInit, layer)
        return layer

    def buildConvTransposeLayers(self, batchNorm, kernelInit, inLayer):
        layer = inLayer
        for f in self.convTransposeFilters:
            layer = self.buildConvTransposeLayer(f, batchNorm, kernelInit, layer)
        return layer

    def train(self, epochs, batchSize, evalFreq):
        if not os.path.exists(self.evalDirectoryName):
            os.makedirs(self.evalDirectoryName)

        batchesPerEpoch = int(self.data.getDatasetShape() / batchSize)
        halfBatch = int(batchSize / 2)

        self.plotStartingImageSamples()

        self.startTime = time.time()

        for i in range(epochs):
            for j in range(batchesPerEpoch):
                xReal, yReal = self.data.generateRealTrainingSamples(halfBatch)
                dRealLoss, dRealAcc = self.discriminator.train_on_batch(xReal, yReal)

                xFake, yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, halfBatch)
                dFakeLoss, dFakeAcc = self.discriminator.train_on_batch(xFake, yFake)

                xGan, yGan = self.data.generateFakeTrainingGanSamples(self.latentDim, batchSize)
                gLoss = self.gan.train_on_batch(xGan, yGan)

                self.realLossHistory.append(dRealLoss)
                self.realAccHistory.append(dRealAcc)
                self.fakeLossHistory.append(dFakeLoss)
                self.fakeAccHistory.append(dFakeAcc)
                self.lossHistory.append(gLoss)

                metrics = ('> %d, %d/%d, dRealLoss=%.3f, dFakeLoss=%.3f, gLoss=%.3f' %
                    (i + 1, j, batchesPerEpoch, dRealLoss, dFakeLoss, gLoss))
                self.metricHistory.append(metrics)
                print(metrics)

            if (i + 1) % evalFreq == 0:
                elaspedTime = f'> elapsed time: {self.getElapsedTime()}'
                self.metricHistory.append(elaspedTime)
                print(elaspedTime)
                self.evaluate(i + 1)

    def evaluate(self, epoch, samples=150):
        xReal, yReal = self.data.generateRealTrainingSamples(samples)
        _, dRealAcc = self.discriminator.evaluate(xReal, yReal)

        xFake, yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)
        _, dFakeAcc = self.discriminator.evaluate(xFake, yFake)

        print('>%d accuracy real: %.0f%%, fake: %.0f%%' % (epoch, dRealAcc * 100, dFakeAcc * 100))

        modelFilename = '%s/generated_model_e%03d.h5' % (self.evalDirectoryName, epoch)
        self.generator.save(modelFilename)

        metricsFilename = '%s/metrics_e%03d.txt' % (self.evalDirectoryName, epoch)
        with open(metricsFilename, 'w') as fd:
            for i in self.metricHistory:
                fd.write(i + '\n')
            self.metricHistory.clear()

        self.plotImageSamples(xFake, epoch)

        self.plotHistory(epoch)

    def plotImageSamples(self, samples, epoch, n=10):
        images, _ = samples
        scaledImages = (images + 1) / 2.0 # scale from -1,1 to 0,1

        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledImages[i, :, :, 0], cmap='gray_r')

        filename = '%s/generated_plot_e%03d.png' % (self.evalDirectoryName, epoch)
        pyplot.savefig(filename)
        pyplot.close()

    def plotStartingImageSamples(self, samples=150):
        xReal, _ = self.data.generateRealTrainingSamples(samples)
        self.plotImageSamples(xReal, -1)

        xFake, _ = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)
        self.plotImageSamples(xFake, 0)

    def plotHistory(self, epoch):
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.realLossHistory, label='dRealLoss')
        pyplot.plot(self.fakeLossHistory, label='dFakeLoss')
        pyplot.plot(self.lossHistory, label='gLoss')
        pyplot.legend()

        pyplot.subplot(2, 1, 2)
        pyplot.plot(self.realAccHistory, label='accReal')
        pyplot.plot(self.fakeAccHistory, label='accFake')
        pyplot.legend()

        pyplot.savefig('%s/loss_acc_history_e%03d.png' % (self.evalDirectoryName, epoch))
        pyplot.close()

    def getElapsedTime(self):
        elapsedTime = time.time() - self.startTime
        return str(datetime.timedelta(seconds=elapsedTime))

    def summary(self):
        print('\nDiscriminator\n')
        self.discriminator.summary()
        
        print('\nGenerator\n')
        self.generator.summary()
        
        print('\nGAN\n')
        self.gan.summary()

if __name__ == '__main__':
    model = ConditionalGAN((28,28,1), 7, 50, 100)
    model.summary()