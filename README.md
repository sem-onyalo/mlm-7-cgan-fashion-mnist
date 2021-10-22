# Conditional GAN (cGAN) MNIST Fashion
A conditional GAN that learns to generate MNIST grayscale fashion images.

 ![Training plots](training-plots.gif)

## Install

```
python -m venv env

.\env\Scripts\activate

pip install -r requirements.txt
```

## View Architecture

```
python main.py --summary
```

## Train

```
python main.py
```

## Generate Images

```
python main.py --modelPath <path-to-model>
```

## Reference

https://machinelearningmastery.com/generative_adversarial_networks/
