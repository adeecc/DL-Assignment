# CS F425 Deep Learning Assignment 1


## Authors
1. Aditya Chopra (2019A7PS0178H)
2. Omkar Pitale (2019A7PS0083H)
3. Aditya Jhaveri (2018A7PS0209H)

## Experimental Results for Refernces

> The results are the mean values of the metrics, taken over 5 runs. Experimentatl Conditions:

| Hyperparameter |       Value        |
| :------------- | :----------------: |
| Optimizer      |       AdamW        |
| Loss Function  | Cross Entropy Loss |
| Batch Size     |         64         |
| Learning Rate  |  $3\cdot10^{-4}$   |
| Weight Decay   |  $5\cdot10^{-4}$   |

> Unless otherwise specified, every layer (except the last) is followed by ReLU activation. The last layer has implict softmax activation

| Model Name                      | Test Accuracy | Precision | Recall |  Loss  |
| :------------------------------ | :-----------: | :-------: | :----: | :----: |
| [10]                            |    0.8401     |  0.8358   | 0.8386 | 0.4566 |
| [256, sigmoid, 10]              |    0.8573     |   0.856   | 0.8559 | 0.4025 |
| [256, tanh, 10]                 |    0.8723     |  0.8709   | 0.8718 | 0.3459 |
| [256, 10]                       |     0.883     |  0.8816   | 0.8821 | 0.3306 |
| [512, 10]                       |    0.8828     |  0.8828   | 0.8818 | 0.3268 |
| [2048, 10]                      |    0.8914     |   0.892   | 0.8913 | 0.3094 |
| [350, 350, 10]                  |    0.8906     |  0.8892   | 0.8899 | 0.3119 |
| [400, 128, 10]                  |     0.889     |  0.8876   | 0.8888 | 0.3155 |
| [920, 920, 10]                  |    0.8867     |  0.8875   | 0.8861 | 0.3312 |
| [720, 720, 720, 10]             |    0.8912     |  0.8932   | 0.8919 | 0.3305 |
| [512, 128, 10]                  |    0.8756     |  0.8834   | 0.8771 | 0.3506 |
| [512, 256, 128, 64, 32, 16, 10] |    0.8809     |  0.8811   | 0.8804 | 0.3490 |

# Comparisons and Result Interpretation
## Baseline Model

We compare the performance of all models with respect to the most basic artificial neural network: A Multiclass Logistic Regression Model. 

## Effect of Activation Functions
> Models under consideration
> 1. [256, Sigmoid, 10]
> 2. [256, tanh, 10]
> 3. [256, relu, 10]

The performance of ReLU superseded all others in all test runs, followed by tanh and then sigmoid. The reason suggested by most papers in two fold: 
- Vanishing Gradients: tanh and sigmoid have their gradients saturate to 0, when the outputs reach near their respective extrema. Consequently, the models stop learning, increasing the number of epochs required for training. This problem is especially evident in DNNs and Image Classification. ReLU does not suffer with such problems, and as the output grows the gradient balances it out.

- Sparsity: ReLU produces 0, whenever the inputs are 0. The more such units that exist in a layer the more sparse the resulting representation. Sigmoids on the other hand are always likely to generate some non-zero value resulting in dense representations. Sparse representations seem to be more beneficial than dense representations. 


Owing to the dominating performance of ReLU, we use the same in all further experiments.

## Effect of Number of Parameters
- We keep the architecture constant in our investigation.

> Models under Consideration
> 1. Model 1: [10]
> 2. Model 2: [5, ReLU, 10]
> 3. Model 3: [256, ReLU, 10]
> 4. Model 4: [512, ReLU, 10]
> 5. Model 5: [2048, ReLU, 10]

- In general, models with more parameters have better results.
- The performance of Model 2 is the worst owing to the fact that the comparatively much larger input space is converted to a tiny encoding space, which is not enough to represent the complex features in inputs and hence the extremely poor performance. 
- As the number of parameters increases, more complex features can be represented and better the performance of the model. A smaller parameter space limits the kind of secondary representations that can be generated.


## Effect of Addition of Layers
- We keep total number of parameters as close as possible in all models considered simultaneously

> Models under Consideration
> 1. [512, relu, 10]
> 2. [350, relu, 350, relu, 10]
> 3. [400, relu, 128, relu, 10]
> 4. [2048, relu, 10]
> 5. [920, relu, 920, relu, 10]
> 6. [720, relu, 720, relu, 720, relu, 10]




## Encoder type models
> Models under Consideration
> 1. Baseline: [512, relu, 128, relu, 10]
> 2. Very Small Encoding: [5, relu, 10]
> 3. Many Layers: [512, relu, 256, relu, 128, relu, 64, relu, 32, relu, 10]
> 4. AutoEncoder Type: [256, relu, 64, relu, 128, relu, 10]


## CNNs
- ReLU implied
> 1. [(1, 6, 5), max_pool(2, 2), (6, 16, 10), max_pool(2, 2), 16x4x4, 128, 64, 10]





