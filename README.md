# PyTorch CNN for Text Classification
PyTorch implementation of the small model from the paper "Character-level Convolutional Networks for Text Classification" [(Zhang et al., 2015)](http://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf?spm=a2c4e.11153940.blogcont576283.36.3ac276778WChsu&file=5782-character-level-convolutional-networks-for-text-classification.pdf). On the yelp full review dataset this model achieves 40.01% error rate (41.41% reported in paper). The only difference from the original implementation is the use of the Adam optimizer as opposed to SGD with momentum.

## Requirements
- Python 3.6
- PyTorch 0.4
- Numpy
- Pandas

To test the model, create a directory "data" and unzip the yelp review full dataset from the project's [data repository](https://drive.google.com/drive/u/1/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M) into it. Then run the cells in the notebook.
