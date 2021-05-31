# Generative Locally Linear Embedding (GLLE)

## How to use the module

### About

This open-source module can be used for manifold unfolding and visualization. It generates various unfoldings in an stochastic way where the generated unfoldings are related to the deterministic Locally Linear Embedding (LLE). For more information about the original deterministic LLE, one can refer to our tutorial available at https://arxiv.org/abs/2011.10925.

### Preparation

The main dependencies are: `Python3, numpy, matplotlib, scikit-learn, scikit-image`

Make sure that you are in the root folder of module when running commands. You may need to install the requirements:

```shell
pip install -r requirements.txt
```

### Dataset loading

### Module running

## Theory of GLLE algorithms

This is the open-source code for the GLLE module. 

The theory of this module is explained in detail in the following paper:

- Benyamin Ghojogh, Ali Ghodsi, Fakhri Karray, Mark Crowley. "Generative Locally Linear Embedding", arXiv preprint 	arXiv:2104.01525, 2021.
- Link of paper at arXiv: https://arxiv.org/abs/2104.01525

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
  booktitle={arXiv preprint arXiv:2104.01525, Under review of IEEE SMC conference},
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
