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
| Batch Size     |         32         |
| Learning Rate  |  $3\cdot10^{-4}$   |
| Weight Decay   |  $1\cdot10^{-3}$   |

> Unless otherwise specified, every layer (except the last) is followed by ReLU activation. The last layer has implict softmax activation

| name | model                             | avg_acc | avg_prec |  avg_rec | avg_loss |
| ---: | :-------------------------------- | ------: | -------: | -------: | -------: |
|    0 | [10]                              | 0.83985 |  0.82907 | 0.829585 |  0.44684 |
|    1 | [256, sigmoid, 10]                | 0.85985 | 0.848273 | 0.850084 | 0.384734 |
|    2 | [256, tanh, 10]                   |  0.8786 | 0.867902 | 0.868314 | 0.348831 |
|    3 | [256, relu, 10]                   |  0.8808 | 0.870984 | 0.871783 | 0.335652 |
|    4 | [512, 10]                         |  0.8843 | 0.873946 | 0.876251 | 0.336186 |
|    5 | [350, 350, 10]                    |  0.8836 | 0.873839 | 0.875851 | 0.361303 |
|    6 | [400, 128, 10]                    | 0.88545 | 0.875128 | 0.877718 |  0.34156 |
|    7 | [2048, 10]                        |  0.8869 | 0.878581 | 0.878968 | 0.329515 |
|    8 | [920, 920, 10]                    |  0.8853 | 0.877576 | 0.878074 |  0.34197 |
|    9 | [720, 720, 720, 10]               | 0.88545 |  0.87544 | 0.875918 |  0.35945 |
|   10 | [512, 128, 10]                    | 0.88215 | 0.870438 | 0.872363 | 0.350242 |
|   11 | [5, 10]                           | 0.83285 | 0.819717 | 0.822192 | 0.486148 |
|   12 | [512, 256, 128, 64, 32, 16, 10]   | 0.87915 | 0.869765 | 0.870835 | 0.382352 |
|   13 | [256, 64, 128, 10]                |  0.8856 | 0.875783 | 0.876916 | 0.350485 |
|   14 | [1024, 2048, 2048, 2048, 256, 10] |  0.8756 | 0.864196 | 0.864935 | 0.404199 |
|   15 | Convolutional Neural Net*         | 0.90015 | 0.891862 | 0.892804 | 0.290758 |

> * The architecture details for the convolutional neural network is provided in its section 

# Comparisons and Result Interpretation
## Baseline Model

We compare the performance of all models with respect to the most basic artificial neural network: A Multiclass Logistic Regression Model, i.e. Model 0. 

## Effect of Activation Functions
### Models under consideration  (in increasing order of performance)

| name | model              | avg_acc | avg_prec |  avg_rec | avg_loss |
| ---: | :----------------- | ------: | -------: | -------: | -------: |
|    1 | [256, sigmoid, 10] | 0.85985 | 0.848273 | 0.850084 | 0.384734 |
|    2 | [256, tanh, 10]    |  0.8786 | 0.867902 | 0.868314 | 0.348831 |
|    3 | [256, relu, 10]    |  0.8808 | 0.870984 | 0.871783 | 0.335652 |

The performance of ReLU superseded all others in all test runs, followed by tanh and then sigmoid. The reason suggested by most papers in two fold: 
- Vanishing Gradients: tanh and sigmoid have their gradients saturate to 0, when the outputs reach near their respective extrema. Consequently, the models stop learning, increasing the number of epochs required for training. This problem is especially evident in DNNs and Image Classification. ReLU does not suffer with such problems, and as the output grows the gradient balances it out.

- Sparsity: ReLU produces 0, whenever the inputs are 0. The more such units that exist in a layer the more sparse the resulting representation. Sigmoids on the other hand are always likely to generate some non-zero value resulting in dense representations. Sparse representations seem to be more beneficial than dense representations. 


Owing to the dominating performance of ReLU, we use the same in all further experiments.

## Effect of Number of Parameters
- We keep the architecture constant in our investigation.

### Models under Consideration

| name | architecture | avg_acc | avg_prec |  avg_rec | avg_loss |
| ---: | :----------- | ------: | -------: | -------: | -------: |
|   11 | [5, 10]      | 0.83285 | 0.819717 | 0.822192 | 0.486148 |
|    0 | [10]         | 0.83985 |  0.82907 | 0.829585 |  0.44684 |
|    3 | [256, 10]    |  0.8808 | 0.870984 | 0.871783 | 0.335652 |
|    4 | [512, 10]    |  0.8843 | 0.873946 | 0.876251 | 0.336186 |
|    7 | [2048, 10]   |  0.8869 | 0.878581 | 0.878968 | 0.329515 |


- In general, models with more parameters have better results.
- The performance of Model 11 is the worst owing to the fact that the comparatively much larger input space is converted to a tiny encoding space, which is not enough to represent the complex features in inputs and hence the extremely poor performance. 
- As the number of parameters increases, more complex features can be represented and better the performance of the model. A smaller parameter space limits the kind of secondary representations that can be generated.
- However, as we will note later, very large parameters spaces might also not desired due to diminishing returns. The model requires much more training to achieve similar levels of accuracy, even more so to improve upon others (Eg: Model 14) 

## Effect of Addition of Layers
- We keep total number of parameters as close as possible in all models considered simultaneously

### Models under Consideration

| name | architecture        | avg_acc | avg_prec |  avg_rec | avg_loss |
| ---: | :------------------ | ------: | -------: | -------: | -------: |
|    5 | [350, 350, 10]      |  0.8836 | 0.873839 | 0.875851 | 0.361303 |
|    4 | [512, 10]           |  0.8843 | 0.873946 | 0.876251 | 0.336186 |
|    6 | [400, 128, 10]      | 0.88545 | 0.875128 | 0.877718 |  0.34156 |
|    8 | [920, 920, 10]      |  0.8853 | 0.877576 | 0.878074 |  0.34197 |
|    9 | [720, 720, 720, 10] | 0.88545 |  0.87544 | 0.875918 |  0.35945 |
|    7 | [2048, 10]          |  0.8869 | 0.878581 | 0.878968 | 0.329515 |



## Encoder type models
### Models under Consideration

| name | architecture                      | avg_acc | avg_prec |  avg_rec | avg_loss |
| ---: | :-------------------------------- | ------: | -------: | -------: | -------: |
|   11 | [5, 10]                           | 0.83285 | 0.819717 | 0.822192 | 0.486148 |
|   14 | [1024, 2048, 2048, 2048, 256, 10] |  0.8756 | 0.864196 | 0.864935 | 0.404199 |
|   12 | [512, 256, 128, 64, 32, 16, 10]   | 0.87915 | 0.869765 | 0.870835 | 0.382352 |
|   10 | [512, 128, 10]                    | 0.88215 | 0.870438 | 0.872363 | 0.350242 |
|   13 | [256, 64, 128, 10]                |  0.8856 | 0.875783 | 0.876916 | 0.350485 |

- We make use of hierarchial representations to squeeze out as much data as possible. At each level, we expect the representations to become more complex. 
- For the model 11, as discussed previously the encoding space is too small to form any complex representations, and information is lost leading to poor performance.
- Secondly, we also note that owing to large parameter space in Models 14 and 12, even though the generalization is expected to be good, it takes a long time to train a model to sufficient performance. This effec is more pronounced in model 14 compared to model 12 resulting in worse performance.
- Model 10 uses the best of both worlds, with a sufficiently large parameter space size and number of layers to get more complex representations and better performance than the previous models.
- Model 13 uses an **Autoencoder** type architecture and gives the best performance overall. This can be attributed to the fact that the model is forced to learn a compressive encoding of the input and recreate the feature space. This allows it to abstract away unnecesary details, and utilise the important features for better performance.


## CNNs
### Architecture: 
1. 2D Convolution: 6 filters and 5x5 kernel 
2. ReLU
3. Max Pool: 2x2 Kernel
4. 2D Convolution: 16 filters and 10x10 kernel 
5. ReLU
6. Max Pool: 2x2 Kernel
7. Flattening
8. Fully Connected Layers with hidden sizes: [128, 64, 10]

| name | model                        | avg_acc | avg_prec |  avg_rec | avg_loss |
| ---: | :--------------------------- | ------: | -------: | -------: | -------: |
|   15 | Convolutional Neural Network | 0.90015 | 0.891862 | 0.892804 | 0.290758 |

- The CNN uses the least number of parameters, yet performs consistently better than any other model considered. 
- A CNN makes use of several filters and the convolution operation to extract features from the inputs with increasing sophistication at each layer (such as edges and contours on lower layers, to more complex shapes and patterns). 
- This is particularly well suited to Computer Vision tasks due to kind of features desired and extracted.

