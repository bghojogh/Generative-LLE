# Generative Locally Linear Embedding (GLLE)

## About

This is the open-source code for the GLLE module. This open-source module can be used for manifold unfolding and visualization. It generates various unfoldings in an stochastic way where the generated unfoldings are related to the deterministic Locally Linear Embedding (LLE). 
- For more information about the original deterministic LLE, one can refer to our tutorial available at https://arxiv.org/abs/2011.10925. 
- For information about the theory of stochastic GLLE, one can see our theoretical paper at https://arxiv.org/abs/2104.01525.
- The paper for software module of GLLE is available at https://doi.org/10.1016/j.simpa.2021.100105.

## How to use the module

### Preparation

The main dependencies are: `Python3, numpy, matplotlib, scikit-learn`

Make sure that you are in the root folder of module when running commands. You may need to install the requirements:

```shell
pip install -r requirements.txt
```

The user can choose the settings in the `settings.json` file. The options of settings in that file are as follows:

```
================ method: 
  LLE ---> it is original deterministic LLE
  GLLE ---> it is GLLE with EM algorithm
  GLLE_DirectSampling ---> it is GLLE with direct sampling
================ dataset:
  Swiss_roll, Swiss_roll_hole, S_curve, Sphere, Sphere_small ---> these are ready toy datasets
  User_data ---> User can put their dataset as a csv files named "data" (row-wise data) and an optional "colors" file
================ make_dataset_again:
  True ---> generate the ready toy datasets again (with new possible data settings)
  False ---> load the previously generated toy dataset (it will throw error if you have not generated a dataset before)
================ embed_again:
  True ---> train the embedding (unfolding) again
  False ---> do not train again and load the previous training phase. This can be useful for when user wants to generate several unfoldings and does not want to train again
================ generate_embedding_again:
  True ---> generate [multiple] unfoldings (embeddings)
  False ---> do not generate unfoldings (embeddings)
================ analyze_covariance_scales:
  True ---> generate unfoldings for various scales of covariance matrix for the sake of analysis
  False ---> do not generate unfoldings for various scales of covariance matrix
================ n_generation_of_embedding:
  A positive integer ---> it is the number of unfoldings (embeddings) to generate
================ max_iterations:
  A positive integer ---> maximum number of iterations for EM algorithm in stochastic linear reconstruction of GLLE
================ n_components:
  A positive integer (between 1 and dimensionality of data) ---> the dimensionality of unfolding (embedding)
================ verbosity:
  0 ---> do not print logging information
  1 ---> print logging information of level one
  2 ---> print logging information of levels one and two
```

### Dataset loading

For the toy datasets, if "make_dataset_again" is set to True in settings, the dataset is generated in its appropriate folder in path "./datasets/dataset_name/". 

For the user data, user should put their data and possibly color or labels in a folder named "User_data" in path "./datasets/User_data/". 
The format of user data should be:
```
"data.csv" (mandatory) ---> a row-wise dataset (rows are instances and columns are features)
"color.csv" (optional) ---> a column vector (rows are instances) whose values are the colors of points (for their relative positions in manifold)
"labels.csv" (optional) ---> a column vector (rows are instances) whose values are the labels of points (for belonging to classes)
```

### Module running

After installing the requirements, setting the settings in the json file, and putting the dataset in the User_data folder (or setting dataset to be one of the ready datasets), user should run the following command for running the module (you should be in the root folder of module):
```shell
python main.py
```

## Theory of GLLE algorithms used in module

The theory of this module is explained in detail in the following paper:

- GLLE paper (theory): Benyamin Ghojogh, Ali Ghodsi, Fakhri Karray, Mark Crowley. "Generative Locally Linear Embedding", arXiv preprint arXiv:2104.01525, 2021.
- LLE paper (tutorial): Benyamin Ghojogh, Ali Ghodsi, Fakhri Karray, and Mark Crowley. "Locally Linear Embedding and its Variants: Tutorial and Survey." arXiv preprint arXiv:2011.10925, 2020.
- Link of GLLE paper at arXiv: https://arxiv.org/abs/2104.01525
- Link of LLE tutorial at arXiv: https://arxiv.org/abs/2011.10925

The GLLE algorithms have stochastic linear reconstruction rather than deterministic linear reconstruction. 

## Examples for manifold unfolding by GLLE algorithms

Consider these nonlinear manifolds:

<img src="https://user-images.githubusercontent.com/66282117/113497353-5203f980-94d1-11eb-86f8-1f1b4d86f173.png" width="30%">

Unfolding these manifolds using (a) original LLE, (b) GLLE with EM algorithm, and (c) GLLE with direct sampling:

![GLLE_generations](https://user-images.githubusercontent.com/66282117/113497394-a7400b00-94d1-11eb-9101-6d67b6bfefc4.png)

## Paper/Module citation

If you are using this module, please cite the following papers:

The bib citation for theoretical paper is:
<pre>
@InProceedings{ghojogh2021generative,
  title={Generative Locally Linear Embedding},
  author={Ghojogh, Benyamin and Ghodsi, Ali and Karray, Fakhri and Crowley, Mark},
  booktitle={arXiv preprint arXiv:2104.01525},
  year={2021}
}
</pre>

The bib citation for the software module paper is:
<pre>
@article{ghojogh2021generativeSoftware,
  title={Generative Locally Linear Embedding: A Module for Manifold Unfolding and Visualization},
  author={Ghojogh, Benyamin and Ghodsi, Ali and Karray, Fakhri and Crowley, Mark},
  journal={Software Impacts},
  publisher={Elsevier},
  year={2021}
}
</pre>
