# Person re-identification with Wasserstein GAN

## This is the pytorch version for implementing person re-identification (person re-ID) with ResNet50 model as baseline model and use WGAN, re-ranking and random erasing to improve the performance.

To train it on Market1501 and DukeMTMC-reid, you need to download the datasets from their website, and run train.py and train_duke.py respectively.

To extract the features, run test.py and test_duke.py.

To evaluate the model, run evaluate.py.

We did not put the codes of Wasserstein GAN part here yet.
