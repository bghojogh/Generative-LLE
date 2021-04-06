# Generative Locally Linear Embedding (GLLE)

This is the code for the paper/project:

- Benyamin Ghojogh, Ali Ghodsi, Fakhri Karray, Mark Crowley. "Generative Locally Linear Embedding", arXiv preprint 	arXiv:2104.01525, 2021.
- Link of paper at arXiv: https://arxiv.org/abs/2104.01525

The proposed GLLE algorithms have stochastic linear reconstruction rather than deterministic linear reconstruction. 

## Examples for manifold unfolding by GLLE algorithms:

Consider these nonlinear manifolds:

<img src="https://user-images.githubusercontent.com/66282117/113497353-5203f980-94d1-11eb-86f8-1f1b4d86f173.png" width="30%">

Unfolding these manifolds using (a) original LLE, (b) GLLE with EM algorithm, and (c) GLLE with direct sampling:

![GLLE_generations](https://user-images.githubusercontent.com/66282117/113497394-a7400b00-94d1-11eb-9101-6d67b6bfefc4.png)

