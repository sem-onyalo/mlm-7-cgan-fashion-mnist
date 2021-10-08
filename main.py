import argparse

from data import Data
from model import ConditionalGAN, Inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--latentDim', '-d', type=int, default=100, help='Latent space dimension')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batchsize', '-b', type=int, default=128, help='The training batch size')
    parser.add_argument('--evalfreq', '-v', type=int, default=10, help='Frequency to run model evaluations')
    parser.add_argument('--classes', '-c', type=int, default=10, help='Number of classes in the training data')
    parser.add_argument('--modelPath', '-m', type=str, default=None, help='Path to model to load. If this is set script will run inference instead of training.')
    parser.add_argument('--summary', '-s', action='store_true', help='Output model summary only. If this is set model will not be trained.')
    parser.add_argument('--convFilters', type=str, default='128,128', help='')
    parser.add_argument('--convTransposeFilters', type=str, default='128,128', help='')
    parser.add_argument('--generatorInputFilters', type=int, default=128, help='')
    parser.add_argument('--adamLearningRate', type=float, default=0.0002, help='')
    parser.add_argument('--adamBeta1', type=float, default=0.5, help='')
    parser.add_argument('--kernelInitStdDev', type=float, default=0.02, help='')
    parser.add_argument('--leakyReluAlpha', type=float, default=0.2, help='')
    parser.add_argument('--dropoutRate', type=float, default=0.4, help='')
    args = parser.parse_args()

    dInputShape = (28,28,1)
    dLabelDim = 50
    dImageDim = 7

    data = Data(args.classes)

    if args.modelPath == None:
        model = ConditionalGAN(data, dInputShape, dImageDim, dLabelDim, args.latentDim, args.classes, args)
        if args.summary:
            model.summary()
        else:
            model.train(args.epochs, args.batchsize, args.evalfreq)
    else:
        inference = Inference(data, args.modelPath, args.latentDim)
        inference.run()