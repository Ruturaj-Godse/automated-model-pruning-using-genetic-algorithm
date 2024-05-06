# Automated Model Pruning using Genetic Algorithm

This project presents a simple framework for optimizing deep neural networks through automated pruning using Genetic Algorithm. Our approach aims to reduce model complexity and enhance computational efficiency without significantly compromising accuracy. In contrast to traditional pruning techniques, our approach doesn't use any rule-based method or strategy to prune the weights. Rather, our system uses the genetic algorithm to learn which weights can be pruned and which are critical. The genetic algorithm over generations learns the best pruning mask to be applied on each layer in order to achieve maximum sparsity while maintaining accuracy. We do not need to specify any knowledge about the model or the pruning technique to be used. This framework can be used as a push-the-button solution for significantly pruning any pre-trained model within a few hours. The framework also presents an option to prune each layer of the model independently and thus can be easily scaled for large models using distributed systems. 

## Outline of the Code Repository
├── LICENSE
├── LeNet
│   ├── lenet.py
│   └── lenet_ckpt.pth
├── README.md
├── ResNet
│   ├── resnet.py
│   └── resnet_ckpt.pth
├── plotting.ipynb
├── pruning.py
├── solutions
│   ├── LeNet
│   └── ResNet
├── sparse_models
│   ├── LeNet
│   │   ├── sparse_weights.pth
│   │   └── sparse_weights_retrained.pth
│   └── ResNet
│       ├── sparse_weights.pth
│       └── sparse_weights_retrained.pth
└── utils.py
