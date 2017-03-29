# Fully Connected DenseNets for Semantic Segmentation
Fully Connected DenseNet for Image Segmentation implementation of the paper [The One Hundred Layers Tiramisu : Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326v1.pdf)

# Differences
- Use of SubPixelConvolution instead of Deconvolution as default method for Upsampling.

# Usage :

Simply import the `densenet_fc.py` script and call the create method:

```
import densenet_fc as dc

model = DenseNetFCN((32, 32, 3), nb_dense_block=5, growth_rate=16,
                        nb_layers_per_block=4, upsampling_type='upsampling', classes=1)
```

# Requirements
Keras 1.2.2 
Theano (master branch) / Tensorflow 1.0+
h5py

