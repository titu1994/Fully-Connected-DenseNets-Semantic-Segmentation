# Fully Connected DenseNets for Semantic Segmentation
Fully Connected DenseNet for Image Segmentation implementation of the paper [The One Hundred Layers Tiramisu : Fully Convolutional DenseNets for Semantic Segmentation](https://arxiv.org/pdf/1611.09326v1.pdf)

# Differences
- Use of SubPixelConvolution instead of Deconvolution as default method for Upsampling.

# Usage :

Simply import the `densenet_fc.py` script and call the create method:

```
import densenet_fc as dc

model = dc.create_fc_dense_net(nb_classes=10, img_dim=(3, 224, 224), nb_dense_block=5, growth_rate=12,
                               nb_filter=16, nb_layers=4)
```

# Requirements
Keras 1.2.1+ (only theano backend is working right now).
Theano (master branch)
h5py

