# DSC 261 Final Project: Relational Deep Learning and Explainability of Graph Neural Network
Developing Graph Neural Networks (GNNs) for heterogeneous graphs are overseeing the explainability.

## Installation 
```
pip install -r requirements.txt
```
It is recommended to run on Datahub instance since some of the packages might not be recognized while running on the local machine.

## Model
gnn/captum_explainer.py contains the implementation of GNN model and the explainer for all four attribution methods methods. The remaining folders are helpers.
The file generates visualizations for the feature importance using each of the methods.
Guided Backpropagation: 
![alt text](https://github.com/shreya2031/gnn-explainainability/tree/main/results/feature_importance_ GuidedBackprop.png "Guided Backpropagation")

## Future Work
1. PyTorch Geomtric does not yet support Captum evaluation metrics to evaluate the truthfulness of the explanations for regression tasks. 
2. Relbench is a novel framework, support is limited and it is still in development phase. 
