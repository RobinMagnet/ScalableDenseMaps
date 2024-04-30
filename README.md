# ScalableDenseMaps

[![](https://github.com/RobinMagnet/ScalableDenseMaps/actions/workflows/documentation.yml/badge.svg)](https://robinmagnet.github.io/ScalableDenseMaps/)


This library has two purposes:
 1. It provides a unified representation of correspondences between objects such as surfaces or point clouds. On **both numpy and torch**
 2. It provides a GPU-memory scalable version of such maps when they can be represented by dense matrices, as presented in the [*Memory-Scalable and Simplified Functional Map Learning*](https://arxiv.org/abs/2404.00330) paper.


# Unified Representation

In the case of surfaces $S_1$ and $S_2$ with respectively $n_1$ and $n_2$ vertices, correspondences $T:S_2\to S_1$ are usually represented in one of the following way:
 1. A vertex to vertex map, where $T$ maps each vertex of $S_2$ to a vertex of $S_1$. It is then usually represented as an array (or tensor) `p2p_21` of shape $n_2$, where `p2p_21[i]` gives the index of the target vertex in $S_1$ for vertex $i$ in $S_2$. When writing results, one usually use an equivalent matrix representation $\Pi\in\{0,1\}^{n_2\times n_1}$, where $\Pi_{ij}=1$ iff $T(x_i^2) = x_j^1$ (or `p2p_21[i]=j`).
2. A vertex to point map, where each $T$ maps each vertex of $S_2$ to a point of $S_1$, which can lie on any of the face of $S_1$. One can again use a matrix representation $\Pi\in[0,1]^{n_2\times n_1}$, with $\forall i,\ \sum_j \Pi_{ij} = 1$, and at most $3$ non-zero entry per-line can easily be defined. See [this paper](https://onlinelibrary.wiley.com/doi/full/10.1111/cgf.13254) for more details.
3. A "fuzzy" map $T$, which is represented by a **dense** matrix $\Pi \in[0,1]^{n_2\times n_1}$. For exampe, given embeddings $e_i^1$ and $e_j^2$ on for each vertex of each shape, the softmax map $\Pi = \frac{1}{\sum_j \exp(S_{ij})}\exp(S_{ij})$, with $(S_{ij})_{ij}$ a matrix of scores (or proximity) for all pairs of embeddings $e_i^1$ and $e_j^2$

In any of these case, the exact representation of $\Pi$ is usually not useful, and one usually seeks to perform some operation with these maps such
- Extraction a vertex-to-vertex map for any of these representation
- Pulling back functions $f$ (such as uv coordinates) using $\Pi x$
- Combining maps $\Pi_{13}=\Pi_{12} \Pi_{23}$


This package provides simple wrapper around these representation, in either numpy or torch (all cuda-compatible).

```python
from densemaps.torch import maps

emb1 = # Use some per-vertex embedding for object 1. (N1, p)
emb2 = # Use some per-vertex embedding for object 2. (N2, p)

P21 = maps.KernelDistMap(emb1, emb2, blur=1e-1)  # A "dense" kernel map, not used in memory

# If my embeddings were not on CUDA, I can send them easily and come back to cpu
P21.cuda()
P21.cpu()

uv1 = # Get uv-coordinates on mesh1  (N1, 2)
uv2 = P21 @ uv1  # Transfered uv coordinates (n2, 2)

P21_dense = P21._to_dense() # I can get the (N2, N1) map back
```

 # Citing this work

 If you use this work, please cite

 ```bibtex
@inproceedings{magnetMemoryScalable2024,
  title = {Memory Scalable and Simplified Functional Map Learning},
  booktitle = {2024 {{IEEE}}/{{CVF Conference}} on {{Computer Vision}} and {{Pattern Recognition}} ({{CVPR}})},
  author = {Magnet, Robin and Ovsjanikov, Maks},
  year = {2024},
  publisher = {IEEE},
}
```