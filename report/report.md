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


## Experiments 
| Model Name | Test Accuracy | Precision | Recall | Loss |
| :------ | :-------: | :------: | :-----: | :-----: |
| 2048_relu | 0.8914 | 0.892 | 0.8913 | 0.3094 |
| 256_relu | 0.883 | 0.8816 | 0.8821 | 0.3306 |
| 256_sigmoid | 0.8573 | 0.856 | 0.8559 | 0.4025 |
| 256_tanh | 0.8723 | 0.8709 | 0.8718 | 0.3459 |
| 350_ReLU_350_ReLU | 0.8906 | 0.8892 | 0.8899 | 0.3119 |
| 400_ReLU_128_ReLU | 0.889 | 0.8876 | 0.8888 | 0.3155 |
| 512_ReLU_128_reLU | 0.8756 | 0.8834 | 0.8771 | 0.3506 |
| 512_ReLU_256_ReLU_128_ReLU_64_ReLU_32_ReLU_16_ReLU | 0.8809 | 0.8811 | 0.8804 | 0.3490 |
| 512_relu | 0.8828 | 0.8828 | 0.8818 |0.3268 |
| 720_ReLU_720_ReLU_720_ReLU | 0.8912 | 0.8932 | 0.8919 | 0.3305 |
| 920_ReLU_920_ReLU | 0.8867 | 0.8875 | 0.8861 | 0.3312 |
| baseline_0 | 0.8401 | 0.8358 | 0.8386 | 0.4566 |


