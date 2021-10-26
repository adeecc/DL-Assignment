# CS F425 Deep Learning Assignment 1


## Authors
1. Aditya Chopra (2019A7PS0178H)
2. Omkar Pitale (2019A7PS0083H)
3. Aditya Jhaveri (2018A7PS0209H)

# Models
## Baseline Model

1. 0 layers => Multi-class logistic regression

## Effect of Activation Functions
We keep number of parameters almost constant

1. [256, Sigmoid, 10]
2. [256, tanh, 10]
3. [256, relu, 10]


## Effect of Addition of Layers
- We keep total number of parameters as close as possible in all the models

1. Base: [512, relu, 10]
2. Equal nodes in hidden layers: [350, relu, 350, relu, 10]
3. Progressive decline: [400, relu, 128, relu, 10]

OR?

4. Base2: [2048, relu, 10]
5. [920, relu, 920, relu, 10]
6. [720, relu, 720, relu, 720, relu, 10]


## Encoder type models
1. Baseline: [512, relu, 128, relu, 10]
2. Very Small Encoding: [5, relu, 10]
3. Many Layers: [512, relu, 256, relu, 128, relu, 64, relu, 32, relu, 10]
4. AutoEncoder Type: [256, relu, 64, relu, 128, relu, 10]

## Change Of Loss Function
1. KL Divergence as Loss Function with best performing DNN Model


## CNNs
- ReLU implied
1. Baseline: [(1, 6, 5), max_pool(2, 2), Flatten, flattened_size, 10]
2. Increasing Kernel Size: [(1, 6, 10), max_pool(2, 2), Flatten, flattened_size, 10]
3. Adding More Filters: [(1, 6, 5), max_pool(2, 2), (6, 16, 3), max_pool(2, 2), Flatten, flattened_size, 10]
