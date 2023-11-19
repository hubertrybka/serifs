# SERIFS: design of target-focused libraries by exploring active ligand subspaces encoded with fingerprints

## Authors: Hubert Rybka, Mateusz Iwan, Anton Siomchen, Tomasz Danel, Sabina Smusz

## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Usage](#usage)
* [Data sources and tools](#data-sources-and-tools)

## General info
SELFIES-based Recurrent Neural Network for Interpretation of Fingerprint Space (SERIFS) is novel generative model for construction of target-specific compound libraries. The proposed model functions as a variational autoencoder, encoding molecular fingerprints into compact, 32-dimensional latent space, which can be processed by a predicive search algorithm to identify subspaces of compunds with high biological activity. The model is then capable of decoding the latent space into SELFIES strings with the use of a GRU-based decoder, which enables us to generate libraries far considered biological target. User is able to retrain the latent classifier in order to conduct search on the latent space and generate compounds libraries matching the fingerprint profile of provided training data. A detailed uder manual is attached below. SERIFS is provided under MIT license.

## Setup
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) following the instructions for your operating system.
2. Clone the repository: `git clone'
3. Install environment from the YAML file: `conda env create -n mldd -f environment.yml`

## Usage
### Activate the environment:  
      conda activate mldd

### Prepare the dataset: 

Please note that a dataset od D2 receptor ligands we used in our paper is available to be downloaded from our [dropbox](https://www.dropbox.com/scl/fo/o7pd38t8gnfoz9reqp7ot/h?rlkey=r2dh1sxrnt34bueza4snxn74b&dl=0) or by launching 
      
      get_datasets.sh

In order to reatrain the latent classifier, you have to provide an appropriate dataset. Put the data into pandas.DataFrame object. The dataframe must contain the following columns:  
      
* 'smiles' - SMILES strings of known ligands in canonical format.  
      
* 'fps' - Klekota&Roth or Morgan (radius=2, nBits=2048) fingerprints of the ligands.  
        The fingerprints have to be saved as ordinary pthon lists, in **dense format** (a list of ints designating the indices of **active bits** in the original fingerprint).
        For python scrip to convert sparse molecular fingerprints into dense format, see serifs.utils.finger.sparse2dense().
          
* 'activity' - Activity class (True, False). By default, we define active compounds as those having
        Ki value <= 100nM and inactive as those of Ki > 100nM.
      
Save dataframe to .parquet file:
```
import pandas as pd
df = pd.DataFrame(columns=['smiles', 'fps', 'activity'])

# ... load data into the dataframe
name = '5ht7_ECFP' # example name for the dataset

df.to_parquet(f'data/activity_data/{name}.parquet', index=False)
```
### Train the RNN decoder

(Advanced) This step can be omitted and you can use pretrained models. Model weights, as well as datasets needed for training, 
are available on [dropbox](https://www.dropbox.com/scl/fo/o7pd38t8gnfoz9reqp7ot/h?rlkey=r2dh1sxrnt34bueza4snxn74b&dl=0) and can be batch downloaded using `get_datasets.sh` No more steps are needed to use 
the pretrained model.

If you intend train the RNN, use the following command:

    python train_gru.py

Be sure to edit the config file in advance (config_files/train_config.ini) to set the desired parameters.
Model weigthts and training progress will be saved to models/model_name.

### Train the SVC activity predictor.
Use the following command:
  
    python train_clf.py

Be sure to provide  path to the dataset file (prepared as explained above) using the -d (--data_path) flag.
Other parameters are optional and can be set using command line arguments.
```
--data_path DATA_PATH, -d DATA_PATH  
                        Path to data file (prepared as described in above section)
--c_param C_PARAM, -c C_PARAM
                  C parameter for SVM (default: 50)
                  Commonly a float in the range [0.01, 100]
--kernel KERNEL, -k KERNEL
                  Kernel type for SVM (default: 'rbf')
                  One of: 'linear', 'poly', 'rbf', 'sigmoid'
--degree DEGREE, -deg DEGREE
                  Degree of polynomial kernel
                  Ignored by other kernels
--gamma GAMMA, -g GAMMA
                  Gamma parameter for SVM (default: 'scale')
                  One of: 'scale', 'auto', or a float
```
For more info on the SVC classifier, please refer to [scikit-learn SVC documentation](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).
  
Now a file with the trained model should be saved in the 'models' directory. Locate the directory,
and save path to a model.pkl file created by the training script inside.
    
It should look like this:
        
    models/model_name/model.pkl

### Perform bayesian search on the latent space
  
The trained activity predictor can be used to perform bayesian search on the latent space
in order to identify latent representations of potential novel ligands.
To perform bayesian search on the latent space, use the following command:

    python bayesian_search.py

Be sure to provide the path to the model file using the -m (--model_path) flag, and the desired number of samples to be 
generated using the -n (--n_samples) flag.

    python bayesian_search.py -m models/model_name/model.pkl -n 1000

Other parameters can be set using the command line arguments:
```
-m MODEL_PATH, --model_path MODEL_PATH
                  Path to the saved activity predictor model
-n N_SAMPLES, --n_samples N_SAMPLES
                  Number of samples to generate
-p INIT_POINTS, --init_points INIT_POINTS
                  Number of initial points to sample (default: 8)
-i N_ITER, --n_iter N_ITER
                  Number of iterations to perform (default: 20)
-b BOUNDS, --bounds BOUNDS
                  Bounds for the latent space search (default: 4.0)
-v VERBOSITY, --verbosity VERBOSITY
                  Verbosity: 0 - silent, 1 - normal, 2 - verbose (default 1)
-w N_WORKERS, --n_workers N_WORKERS
                  Number of workers to use. (default: -1 [all available CPU cores])
```
For more info about the bayesian optimization process and the choice of non-default parameters refer to 
[bayesian-optimization README](https://github.com/bayesian-optimization/BayesianOptimization).
  
Results of the search will be saved in 'outputs' directory.
  
Directory 'SVC_{timestamp}' will be created on /results, containing the following files:  
* latent_vectors.csv - latent vectors identified by the search  
* info.txt - information about the search

### Generate compound libraries from found latent vectors

The generated compounds are filtered according to criteria, which can be modified in config_files/pred_config.ini.  

In order to generate a library, run `python predict.py`.  
Provide path to latent_vectors.csv using -d (--data_path) flag, for example:
  
      python predict.py -d results/SVC_{timestamp}/latent_vectors.csv

Other parameters can be set using the command line arguments:
```
-d DATA_PATH, --data_path DATA_PATH
                  Path to data file 
-n N_SAMPLES, --n_samples N_SAMPLES
                  Number of samples to generate for each latent vector. If > 1, the variety of the generated
                  molecules will be increased by using dropout (default = 10)
-c CONFIG, --config CONFIG
                  Path to config file (default: config_files/pred_config.ini)
-m MODEL_PATH, --model_path MODEL_PATH
                  Path to model weights (default: models/GRUv3_ECFP/epoch_150.pt)
-v VERBOSITY, --verbosity VERBOSITY
                  Verbosity level (0 - silent, 1 - progress, 2 - verbose)
-u USE_CUDA, --use_cuda USE_CUDA
                  Use CUDA if available (default: False)
```

As a result, in results/SVC_{timestamp} dir, a new directory preds_{new_timestamp} will be created. This contains the following files:
* predictions.csv, a file containing SMILES of the generated compounds, as well as some calculated molecular properties
  (QED, MW, logP, ring info, RO5 info rtc.)
* imgs directory, in which .png files depicting the structures of the generated compounds are located
* config.ini, a copy of the configuration file used for prediction (incl. filter criteria)

## Data sources and tools
### Data sources:
* CHEMBL32
  ~1M subset of compounds with molecular mass in range of 200-450 Da, with no RO5 violations.
* ZINC-250k

  Datasets are available on [dropbox](https://www.dropbox.com/scl/fo/o7pd38t8gnfoz9reqp7ot/h?rlkey=r2dh1sxrnt34bueza4snxn74b&dl=0)
  and can be batch downloaded using `get_datasets.sh`

