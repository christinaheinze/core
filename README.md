# CoRe: Conditional Variance Penalties and Domain Shift Robustness
TensorFlow implementation of 'CoRe' (COnditional Variance REgularization), proposed in ["Conditional Variance Penalties and Domain Shift Robustness"](https://arxiv.org/abs/1710.11469).

## Method
The aim is to build classifiers that are robust against specific interventions. These domain-shift interventions are defined in a causal graph, extending the framework of Gong et al (2016). In contrast to Gong et al. we work on a setting where the domain variable itself is latent but we can observe for some instances a so-called identifier variables that indicates, for example, presence of the same person or object across different images.
Penalizing the variance across instances that share the same class label and identifier leads to robustness against strong domain-shift interventions.

## Software

### Requirements
* Python 3
* TensorFlow version >1.4


### Reproducing examples
Running the following command reproduces the example 2 from the manuscript:
```
sh examples/submit-nonlinear-core.sh
```
The pooled estimator can be run with:
```
sh examples/submit-nonlinear-baseline.sh
```

For the rotated MNIST example from section 7.5, the respective files are
``examples/submit-rotmnist-core.sh`` and ``examples/submit-rotmnist-baseline.sh``. We will be adding more code to reproduce the other experiments shown in the manuscript.

## References
C. Heinze-Deml and N. Meinshausen. "Conditional Variance Penalties and Domain Shift Robustness". [arXiv](https://arxiv.org/abs/1710.11469).

M. Gong, K. Zhang, T. Liu, D. Tao, C. Glymour, and B. Schoelkopf. Domain adaptation with conditional transferable components. In International Conference on Machine Learning, 2016.
