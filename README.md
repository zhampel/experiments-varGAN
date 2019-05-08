# Variational explorations of GANs

Experimenting with variational inference in GANs


## Requirements

The package as well as the necessary requirements can be installed by running `make` or via
```
virtualenv -p /usr/local/bin/python3 venv
source venv/bin/activate
python setup.py install
```


## Generate Data

To generate some n-dimensional Gaussian data, use the `gen-samples.py` script:
```
python gen-samples.py -n 1000 -b 10 -d 1 2 10
```

where `-n` gives the samples per batch, `-b` is the number of batches to run
and `-d` is a list to specify the dimensions n of multi-dimensional Gaussians datasets to generate.

This will create a directory `datasets` with the saved generated data named according to the dimensionality,
along with figures displaing the Euclidean norm distribution of the datasets:
```
datasets/
│   data_dim1.h5
│   data_dim2.h5
│   data_dim10.h5
│   hist_dim1.png
│   hist_dim2.png
│   hist_dim10.png
```


## Train a Generator

To train the generator on a specific n-dimensional Gaussian dataset, use
```
python train.py -d 10 -n 100 -s mnist
```
where `-d` specifies the dataset to model, `-n` is the number of epochs over which to train,
and `-s` is used to specify the image dataset used as the latent space.
The two current options for the latent space datasets are MNIST & Fashion-MNIST.

This will save results from the training in the following directory structure:
```
runs/
│
└───dim10/
    │    training_details.csv
    │    training_model_losses.png
    │    training_pvalues.png
    └────models/
         │    generator.pth.tar
         │    discriminator.pth.tar
    └────samples/
         │    hist_epoch00000.png
         │        ...
         │    hist_epoch00100.png
```
