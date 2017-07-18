# Doublet filtering
Neural network model to filter track seeds based on their hit shapes and positions in the detector.


# Project overview
- **make_dataset.py** is a small script to split the data into training, validation and test set.
- **dataset.py** contains the Dataset class responsible to load the dataset from disk and do some basic preprocessing to prepare the input for the neural network (e.g. balance the data, reshape the input).
- **doublet_model.py** is the main script responsible for the doublet model training.
- **hyperparam_search.py** performs a grid search over the network's hyperparameters using doublet_model.py
- **random_search.py** performs a random search over the network's hyperparameters using doublet_model.py


# Usage
- data is stored in /eos/cms/store/cmst3/group/dehep/convPixels/
- create the dataset with the command `python make_dataset.py`
- start the training with the command `python doublet_model.py`
- to perform a model selection it is possible to use either `python hyperparam_search.py` or `python random_search.py`

The network hyperparameters can be passed via command line arguments. To see the list of arguments use the command `python doublet_model.py --help`.

The parameter grid for random_search.py must be changed directly on the code modifying the dists parameter. `dists` is a dictionary wich contains doublet_model command-line arguments as keys and list of values to sample in the search as values for each key.
