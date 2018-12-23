# PyTorch CNN for Text Classification
PyTorch implementation of the small model from the paper "Character-level Convolutional Networks for Text Classification" (Zhang et al., 2015). On the yelp full review dataset this model achieves 40.01% error rate (41.41% reported in paper). The only difference from the original implementation is the use of the Adam optimizer as opposed to SGD with momentum.

## Requirements
Python 3.6
PyTorch 0.4
