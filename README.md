# Automated Model Pruning using Genetic Algorithm

This project presents a simple framework for optimizing deep neural networks through automated pruning using Genetic Algorithm. Our approach aims to reduce model complexity and enhance computational efficiency without significantly compromising accuracy. In contrast to traditional pruning techniques, our approach doesn't use any rule-based method or strategy to prune the weights. Rather, our system uses the Genetic Algorithm to learn which weights can be pruned and which are critical. The Genetic Algorithm over generations learns the best pruning mask to be applied on each layer in order to achieve maximum sparsity while maintaining accuracy. We do not need to specify any knowledge about the model or the pruning technique to be used. This framework can be used as a push-the-button solution for significantly pruning any pre-trained model within a few hours. The framework also presents an option to prune each layer of the model independently and thus can be easily scaled for large models using distributed systems. 

## Outline of the Code Repository
```
├── LeNet
│   ├── lenet.py: Python script implementing LeNet-5 model using PyTorch
│   └── lenet_ckpt.pth: Model file storing weights of LeNet-5 model trained on MNIST dataset
├── ResNet
│   ├── resnet.py: Python script implementing ResNet-18 model using PyTorch
│   └── resnet_ckpt.pth: Model file storing weights of ResNet-18 model trained on CIFAR-10 dataset
├── plotting.ipynb: Jupyter Notebook plotting graphs to visualize the obtained results
├── pruning.py (the main file): Python script implementing the logic for automated pruning of neural networks using Genetic Algorithm
├── solutions: Folder storing the solutions (best prune masks) for LeNet and ResNet models
├── sparse_models: Folder storing weights of LeNet and ResNet models after applying the prune masks and retraining
└── utils.py: Python script implementing the utility functions required by the main code
```
