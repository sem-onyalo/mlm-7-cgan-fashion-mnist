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
    def __init__(self, data:Data, inputShape, imageDim, labelDim, latentDim, classes=10) -> None:
        self.data = data
        self.latentDim = latentDim
        self.discriminator = self.createDiscriminator(inputShape, labelDim, classes)
        self.generator = self.createGenerator(latentDim, imageDim, labelDim, classes)
        self.gan = self.createGan()

        self.realLossHistory = list()
        self.realAccHistory = list()
        self.fakeLossHistory = list()
        self.fakeAccHistory = list()
        self.lossHistory = list()
        self.metricHistory = list()

        self.evalDirectoryName = 'eval'

    def createDiscriminator(self, inputShape, labelDim, classes, batchNorm=True) -> Model:
        labelInputNodes = inputShape[0] * inputShape[1]
        init = RandomNormal(stddev=0.02)

        labelInput = Input(shape=(1,))
        labelEmbedding = Embedding(classes, labelDim)(labelInput)
        labelDense = Dense(labelInputNodes)(labelEmbedding)
        labelShaped = Reshape((inputShape[0], inputShape[1], 1))(labelDense)

        imageInput = Input(shape=inputShape)
        imageLabelConcat = Concatenate()([imageInput, labelShaped])

        conv1 = Conv2D(64, (4,4), (2,2), padding='same', kernel_initializer=init)(imageLabelConcat)
        if batchNorm:
            conv1 = BatchNormalization()(conv1)
        actv1 = LeakyReLU(0.2)(conv1)
        drop1 = Dropout(0.4)(actv1)

        conv2 = Conv2D(128, (4,4), (2,2), padding='same', kernel_initializer=init)(drop1)
        if batchNorm:
            conv2 = BatchNormalization()(conv2)
        actv2 = LeakyReLU(0.2)(conv2)
        drop2 = Dropout(0.4)(actv2)

        flattenLayer = Flatten()(drop2)
        outputLayer = Dense(1, activation='sigmoid')(flattenLayer)
        model = Model([imageInput, labelInput], outputLayer)

        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    def createGenerator(self, latentDim, imageDim, labelDim, classes, batchNorm=True) -> Model:
        labelInputNodes = imageDim * imageDim
        init = RandomNormal(stddev=0.02)

        labelInput = Input(shape=(1,))
        labelEmbedding = Embedding(classes, labelDim)(labelInput)
        labelDense = Dense(labelInputNodes, kernel_initializer=init)(labelEmbedding)
        labelShaped = Reshape((imageDim, imageDim, 1))(labelDense)

        imageInputFilters = 256
        imageInputNodes = imageInputFilters * imageDim * imageDim
        imageInput = Input(shape=(latentDim,))
        imageDense = Dense(imageInputNodes, kernel_initializer=init)(imageInput)
        imageActv = ReLU()(imageDense)
        imageShaped = Reshape((imageDim, imageDim, imageInputFilters))(imageActv)

        imageLabelConcat = Concatenate()([labelShaped, imageShaped])

        conv1 = Conv2DTranspose(128, (4,4), (2,2), padding='same', kernel_initializer=init)(imageLabelConcat)
        if batchNorm:
            conv1 = BatchNormalization()(conv1)
        actv1 = ReLU()(conv1)
        drop1 = Dropout(0.4)(actv1)

        conv2 = Conv2DTranspose(128, (4,4), (2,2), padding='same', kernel_initializer=init)(drop1)
        if batchNorm:
            conv2 = BatchNormalization()(conv2)
        actv2 = ReLU()(conv2)
        drop2 = Dropout(0.4)(actv2)

        outputLayer = Conv2D(1, (7,7), padding='same', activation='tanh', kernel_initializer=init)(drop2)
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

        opt = Adam(learning_rate=0.002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    def train(self, epochs, batchSize, evalFreq):
        if not os.path.exists(self.evalDirectoryName):
            os.makedirs(self.evalDirectoryName)

        batchesPerEpoch = int(self.data.dataset.shape[0] / batchSize)
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

                metrics = ('>%d, %d/%d, dRealLoss=%.3f, dFakeLoss=%.3f, gLoss=%.3f' %
                    (i + 1, j, batchesPerEpoch, dRealLoss, dFakeLoss, gLoss))
                self.metricHistory.append(metrics)
                print(metrics)

            if (i + 1) % evalFreq == 0:
                self.evaluate(i + 1)
                self.printElapsedTime()

    def evaluate(self, epoch, samples=150):
        xReal, yReal = self.data.generateRealTrainingSamples(samples)
        _, dRealAcc = self.discriminator.evaluate(xReal, yReal)

        xFake, yFake = self.data.generateFakeTrainingSamples(self.generator, self.latentDim, samples)
        _, dFakeAcc = self.discriminator.evaluate(xFake, yFake)

        print('>%d accuracy real: %.0f%%, fake: %.0f%%' % (epoch, dRealAcc * 100, dFakeAcc * 100))

        modelFilename = '%s/generated_model_e%03d.h5' % self.evalDirectoryName, epoch
        self.generator.save(modelFilename)

        metricsFilename = '%s/metrics_e%03d.txt' % self.evalDirectoryName, epoch
        with open(metricsFilename, 'w') as fd:
            for i in self.metricHistory:
                fd.write(i + '\n')
            self.metricHistory = list()

        self.plotImageSamples(xFake, epoch)

        self.plotHistory(epoch)

    def plotImageSamples(self, samples, epoch, n=10):
        scaledSamples = (samples + 1) / 2.0 # scale from -1,1 to 0,1

        for i in range(n * n):
            pyplot.subplot(n, n, i + 1)
            pyplot.axis('off')
            pyplot.imshow(scaledSamples[i, :, :, 0], cmap='gray_r')

        filename = '%s/generated_plot_e%03d.png' % self.evalDirectoryName, epoch
        pyplot.savefig(filename)
        pyplot.close()

    def plotStartingImageSamples(self, samples=150):
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

        pyplot.savefig('%s/loss_acc_history_e%03d.png' % self.evalDirectoryName, epoch)
        pyplot.close()

    def printElapsedTime(self):
        elapsedTime = time.time() - self.startTime
        print('Elapsed time:', str(datetime.timedelta(seconds=elapsedTime)))

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