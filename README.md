# Automated Model Pruning using Genetic Algorithm

This project presents a simple framework for optimizing deep neural networks through automated pruning using Genetic Algorithm. Our approach aims to reduce model complexity and enhance computational efficiency without significantly compromising accuracy. In contrast to traditional pruning techniques, our approach does not make any assumptions regarding the importance of weights or use any rule-based method to prune the weights. Rather, our system uses the Genetic Algorithm to learn which weights can be pruned and which are critical. The Genetic Algorithm over generations learns the best pruning mask to be applied on each layer in order to achieve maximum sparsity while maintaining accuracy. We do not need to specify any knowledge about the model or the pruning technique to be used. Moreover, this framework can be used as a push-the-button solution for significantly pruning any pre-trained model within a few hours. The framework also presents an option to prune each layer of the model independently and thus can be easily scaled for large models using distributed systems. 

## Outline of the Code Repository
<pre>
├── <b>LeNet</b>
│   ├── <b>lenet.py</b>: Python script implementing LeNet-5 model using PyTorch
│   └── <b>lenet_ckpt.pth</b>: Model file storing weights of LeNet-5 model trained on MNIST dataset
├── <b>ResNet</b>
│   ├── <b>resnet.py</b>: Python script implementing ResNet-18 model using PyTorch
│   └── <b>resnet_ckpt.pth</b>: Model file storing weights of ResNet-18 model trained on CIFAR-10 dataset
├── <b>plotting.ipynb</b>: Jupyter notebook plotting graphs to visualize the obtained results
├── <b>pruning.py (the main file)</b>: Python script implementing the logic for automated pruning of neural networks using Genetic Algorithm
├── <b>solutions</b>: Folder storing the solutions (best prune masks) for LeNet and ResNet models
├── <b>sparse_models</b>: Folder storing weights of LeNet and ResNet models after applying the prune masks and retraining
└── <b>utils.py</b>: Python script implementing the utility functions required by the main code
</pre>


## How to Run the Code
Make sure you are working on a system having a GPU with CUDA installed.

1. To prune a specific layer of the model:
   ```
   python pruning.py <model> --layer <layer name>
   ```
   For example:
   ```
   python pruning.py --resnet --layer module.layer4.0.conv2
   ```
   ```
   python pruning.py --lenet --layer module.classifier.3
   ```
2. To prune all the layers of a model
   ```
   python pruning.py <model>
   ```
   For example:
   ```
   python pruning.py --lenet
   ```
3. To further fine-tune a pruned model:
   ```
   python pruning.py --finetune
   ```
   (currently only ResNet-18 is supported for fine-tuning, LeNet-5 doesn't require further fine-tuning)


**Model and Layer Names:**
```
1. lenet
   -  module.feature.0
   -  module.feature.1
   -  module.feature.2
   -  module.feature.3
   -  module.feature.4
   -  module.feature.5
   -  module.classifier.0
   -  module.classifier.1
   -  module.classifier.2
   -  module.classifier.3
   -  module.classifier.4
   -  module.classifier.5
2. resnet
   -  module.conv1
   -  module.bn1
   -  module.layer1.0.conv1
   -  module.layer1.0.bn1
   -  module.layer1.0.conv2
   -  module.layer1.0.bn2
   -  module.layer1.0.shortcut
   -  module.layer1.1.conv1
   -  module.layer1.1.bn1
   -  module.layer1.1.conv2
   -  module.layer1.1.bn2
   -  module.layer1.1.shortcut
   -  module.layer2.0.conv1
   -  module.layer2.0.bn1
   -  module.layer2.0.conv2
   -  module.layer2.0.bn2
   -  module.layer2.0.shortcut.0
   -  module.layer2.0.shortcut.1
   -  module.layer2.1.conv1
   -  module.layer2.1.bn1
   -  module.layer2.1.conv2
   -  module.layer2.1.bn2
   -  module.layer2.1.shortcut
   -  module.layer3.0.conv1
   -  module.layer3.0.bn1
   -  module.layer3.0.conv2
   -  module.layer3.0.bn2
   -  module.layer3.0.shortcut.0
   -  module.layer3.0.shortcut.1
   -  module.layer3.1.conv1
   -  module.layer3.1.bn1
   -  module.layer3.1.conv2
   -  module.layer3.1.bn2
   -  module.layer3.1.shortcut
   -  module.layer4.0.conv1
   -  module.layer4.0.bn1
   -  module.layer4.0.conv2
   -  module.layer4.0.bn2
   -  module.layer4.0.shortcut.0
   -  module.layer4.0.shortcut.1
   -  module.layer4.1.conv1
   -  module.layer4.1.bn1
   -  module.layer4.1.conv2
   -  module.layer4.1.bn2
   -  module.layer4.1.shortcut
   -  module.linear
```

## Results

We ran our pruning algorithm on the LeNet-5 model using the MNIST dataset and the ResNet-18 model using the CIFAR-10 dataset.

### LeNet-5
The Genetic Algorithm was run for 25 generations (per layer) taking around 15 minutes (per layer) to come up with the best prune mask and fine-tune the model.

![sparsity_vs_no_of_layers_pruned_lenet.png](https://github.com/Ruturaj-Godse/automated-model-pruning-using-genetic-algorithm/blob/main/results/sparsity_vs_no_of_layers_pruned_lenet.png)

The algorithm was able to generate 94.43% sparsity in the LeNet-5 model while maintaining the accuracy at 97.9%.

![sparsity_per_layer_lenet.png](https://github.com/Ruturaj-Godse/automated-model-pruning-using-genetic-algorithm/blob/main/results/sparsity_per_layer_lenet.png)

The algorithm was able to prune each layer by more than 93%. 

### ResNet-18
The Genetic Algorithm was run for 15 generations (per layer) taking around 10 minutes (per layer) to come up with the best prune mask and fine-tune the model. The algorithm was run for two iterations, compressing each layer to 60% in the first iteration and then to as much as possible in the second iteration.

![sparsity_vs_no_of_layers_pruned_resnet_two_iter.png](https://github.com/Ruturaj-Godse/automated-model-pruning-using-genetic-algorithm/blob/main/results/sparsity_vs_no_of_layers_pruned_resnet_two_iter.png)

The algorithm was able to generate 78.91% sparsity in the ResNet-18 model while maintaining the accuracy at 89.62%.

![sparsity_per_layer_resnet_2_iter.png](https://github.com/Ruturaj-Godse/automated-model-pruning-using-genetic-algorithm/blob/main/results/sparsity_per_layer_resnet_2_iter.png)

We can see significant variation in the amount of sparsity achieved in each layer, showing that each layer is not compressable by the same amount. It's best not to make any assumptions regarding the importance of weights while designing the pruning algorithm. Our algorithm is able to learn the importance of weights completely on its own without any bias towards magnitude or correlation, any domain knowledge, or any kind of input from the user.

Comparison with state-of-the-art methods:
| Method            | Accuracy %      | Parameters (in millions)     | Sparsity % |
|-------------------|:---------------:|:----------------------------:|:----------:|
| PIY            | 91.23           | 6.11                         | 45.31      |
| CACP          | 92.03           | 3.50                         | 68.66      |
| P-SLR        | 90.37           | 1.34                         | 88.02      |
| PCNN         | 96.38           | 3.80                         | 66.01      |
| CRL               | 90.97           | 2.52                         | 77.46      |
| AMPGA (ours)         | 90.01           | 2.83                         | 74.68      |
| **AMPGA 2-iter (ours)**  | **89.62**           | **2.35**                        | **78.91**      |

