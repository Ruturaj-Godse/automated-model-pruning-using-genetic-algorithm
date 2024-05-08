'''Main logic for automated pruning of neural networks using genetic algorithm.'''

import argparse
import copy
from heapq import heapify, heappush, heappop
import pickle
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.profiler
import torchvision
import torchvision.transforms as transforms

from utils import progress_bar, get_accuracy, get_module_by_name, get_module_names
from ResNet.resnet import ResNet18
from LeNet.lenet import LeNet


args = None             # command line arguments
solution_mask = None    # pruning mask to be shared with the custom pruning function 

def load_data_cifar10(path='./datasets/cifar10/', num_workers=4):
    """
    Loads and prepares the CIFAR-10 dataset for model training and evaluation. 
    This function splits the dataset into training, validation, and test sets. 
    It applies transformations to the data to enhance model generalizability and performance.

    Parameters:
        path (str): The directory where the CIFAR-10 dataset is stored or will be downloaded.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing three DataLoader objects:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    """

    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=num_workers)

    val_test_set = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)

    val_size = int(0.5 * len(val_test_set))
    test_size = len(val_test_set) - val_size
    val_set, test_set = torch.utils.data.random_split(dataset=val_test_set, lengths=[val_size, test_size])

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def load_data_mnist(path="./datasets/mnist/", num_workers=4):
    """
    Loads and prepares the MNIST dataset for model training and evaluation. 
    This function splits the dataset into training, validation, and test sets. 
    It applies transformations to the data to enhance model generalizability and performance.

    Parameters:
        path (str): The directory where the MNIST dataset is stored or will be downloaded.
        num_workers (int): The number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing three DataLoader objects:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    """

    print('==> Preparing data..')
    train_dataset = torchvision.datasets.mnist.MNIST(root=path, train=True, download=True, transform=transforms.ToTensor())
    val_test_dataset = torchvision.datasets.mnist.MNIST(root=path, train=False, download=True, transform=transforms.ToTensor())

    imgs = torch.stack([img for img, _ in train_dataset], dim=0)

    mean = imgs.view(1, -1).mean(dim=1)    
    std = imgs.view(1, -1).std(dim=1)

    mnist_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    train_dataset = torchvision.datasets.mnist.MNIST(root=path, train=True, download=False, transform=mnist_transforms)
    val_test_dataset = torchvision.datasets.mnist.MNIST(root=path, train=False, download=False, transform=mnist_transforms)

    val_size = int(0.5 * len(val_test_dataset))
    test_size = len(val_test_dataset) - val_size
    val_set, test_set = torch.utils.data.random_split(dataset=val_test_dataset, lengths=[val_size, test_size])

    BATCH_SIZE = 32

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def train(net, train_loader, val_loader, optimizer, criterion, epochs, device):
    """
    Trains a neural network using the specified data loaders, optimizer, loss function, and device.
    The function handles the training loop, including forward pass, loss computation, backpropagation, 
    and parameter updates. It also evaluates the model on a validation set after each epoch to monitor
    performance improvements.

    Parameters:
        net (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer used for adjusting the weights of the model.
        criterion (torch.nn.Module): Loss function used to evaluate the goodness of the model.
        epochs (int): Number of epochs to train the model.
        device (torch.device or str): Device on which to train the model, e.g., 'cuda' or 'cpu'.

    Returns:
        None. The function outputs training and validation results directly to the console and updates 
        the model `net` in place.
    """

    for epoch in range(epochs):

        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        test_acc = get_accuracy(net, val_loader, device)    
        print(f"Val Accuracy: {test_acc}")

    print()


class CustomPruningMethod(prune.BasePruningMethod):
    """
    A custom pruning method that extends PyTorch's BasePruningMethod to implement
    an unstructured pruning technique using a solution mask provided.

    Attributes:
        PRUNING_TYPE (str): Defines the type of pruning as 'unstructured'. This means
            the pruning is not restricted to any particular structure like channels or
            layers, but can occur at individual weight levels across the model.

    Methods:
        compute_mask(t, default_mask):
            Computes a new mask for the tensor 't' using a globally defined 'solution_mask'
            that specifies which elements of the tensor to prune.
    """

    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        """
        Computes and applies a custom pruning mask to the given tensor.
        
        Parameters:
            t (torch.Tensor): The tensor to be pruned.
            default_mask (torch.Tensor): The default binary mask provided by the pruning method.
        
        Returns:
            torch.Tensor: A new mask tensor that has been customized based on the global 'solution_mask'.
        """

        global solution_mask
        mask = torch.reshape(solution_mask, t.shape)
        mask = mask.to('cuda')
        return mask
    
    
def custom_unstructured(module, name):
    """
    Applies the CustomPruningMethod to a specific module of a neural network. 
    This function allows for the unstructured pruning of the module's specified 
    parameter (typically weights) using a globally defined pruning mask.

    Parameters:
        module (torch.nn.Module): The module from a neural network whose parameter 
                                  is to be pruned.
        name (str): The name of the parameter within the module to prune, e.g., 'weight'.

    Returns:
        torch.nn.Module: The same module with the specified parameter now subjected to 
                         the custom pruning process. This allows for in-place modification
                         and reusability of the module in further operations or training.
    """
    CustomPruningMethod.apply(module, name)
    return module


def formula(sparsity, accuracy):
    """
    Computes the objective function value for a given sparsity and accuracy of a pruned neural network. 
    This function calculates a weighted sum of sparsity and accuracy to evaluate the trade-off 
    between model complexity (as measured by sparsity) and model performance (as measured by accuracy).

    Parameters:
        sparsity (float): The sparsity of the model, representing the proportion of the model's 
                          weights that have been pruned, typically a value between 0 and 1.
        accuracy (float): The accuracy of the model on a validation or test dataset, typically 
                          a value between 0 and 1.

    Returns:
        float: The computed value of the objective function, representing the trade-off between 
               sparsity and accuracy.

    The function uses a fixed weight `alpha` of 0.8 to prioritize sparsity, but this can be adjusted 
    depending on specific requirements or preferences for the balance between sparsity and accuracy.
    """
    alpha = 0.8
    return alpha * sparsity + (1 - alpha) * accuracy


def objective_function(model, layer_name, solution, test_loader, accuracy_lower_limit=0, ret=0):
    """
    Evaluates the performance of a pruned model using a specified pruning solution and 
    computes the objective function based on the sparsity and accuracy of the model. This 
    function applies the pruning, tests the pruned model, calculates its sparsity, and 
    uses these metrics to compute the objective function value.

    Parameters:
        model (torch.nn.Module): The original neural network model before pruning.
        layer_name (str): The name of the layer to which the pruning will be applied.
        solution (np.array): A binary array representing the pruning mask where 1 indicates 
                             that the corresponding weight is kept, and 0 indicates it is pruned.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset to evaluate 
                                                   the model's accuracy after pruning.
        accuracy_lower_limit (float, optional): The minimum accuracy threshold required for the 
                                                model to be considered valid. Defaults to 0.
        ret (int, optional): Control flag to determine the type of return value. If 1, the function 
                             returns a tuple of sparsity and accuracy; otherwise, it returns the 
                             computed objective function value. Defaults to 0.

    Returns:
        float or tuple: Depending on the value of `ret`, this function returns either:
                        - A float representing the computed objective function value if `ret` is 0.
                        - A tuple of (sparsity, accuracy) if `ret` is 1.
                        If the model's accuracy is below the `accuracy_lower_limit`, the function 
                        returns 0 to indicate failure to meet the minimum criteria.
    """
    global solution_mask
    temp_model = copy.deepcopy(model)
    solution_mask = torch.tensor(solution)
    layer = get_module_by_name(temp_model, layer_name)
    custom_unstructured(layer, name='weight')
    acc = get_accuracy(temp_model, test_loader, 'cuda')
    spar = (solution.size - np.count_nonzero(solution))/solution.size
    if ret == 1:
        return (spar, acc)
    if acc < accuracy_lower_limit:
        return 0
    return formula(spar, acc)


class GeneticPruning:
    """
    Implements genetic algorithm for pruning neural networks. This class handles the creation
    of an initial population of pruning masks, the evaluation of these masks based on network performance,
    and the evolution of the population over generations to optimize network sparsity while maintaining
    or improving accuracy. It provides methods to prune one or more layers of the model using the above approach.

    Attributes:
        model (torch.nn.Module): The neural network model to be pruned.
        model_name (str): Name of the model, automatically derived from the model's class.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        test_loader (torch.utils.data.DataLoader): DataLoader for test data.
        optimizer (torch.optim.Optimizer): Optimizer used for training the pruned model.
        criterion (torch.nn.Module): Loss function used during training.
        post_prune_epochs (int): Number of epochs to train the model after each pruning iteration.
        post_prune_epochs_per_layer (int): Number of epochs to to train the model after pruning each layer.
        device (torch.device): Device on which to perform computations (e.g., 'cuda' or 'cpu').
        module_names ([str]): List of all module names in the model.
        layer_name (str): Name of the current layer being pruned.
        layer_solutions (dict): Dictionary storing solutions (best prune masks) for each layer.
        accuracy_lower_limit (float): Minimum acceptable accuracy for the pruned model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, criterion, device, post_prune_epochs, post_prune_epochs_per_layer, accuracy_lower_limit) -> None:
        self.model = model
        self.model_name = model.module.__class__.__name__
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.post_prune_epochs = post_prune_epochs
        self.post_prune_epochs_per_layer = post_prune_epochs_per_layer
        self.device = device
        self.module_names = []
        get_module_names(self.model, '', self.module_names)
        self.layer_name = None
        self.layer_solutions = {}
        self.accuracy_lower_limit = accuracy_lower_limit

    
    def objective_function_wrapper(self, solution, ret=0):
        """
        Wrapper method for the objective_function to evaluate the efficacy of a pruning solution based 
        on model sparsity and accuracy using the validation dataset. This method allows for easy integration 
        of the objective function within the genetic algorithm workflow by automatically passing the relevant 
        model, layer, and data loader.

        Parameters:
            solution (np.array): The pruning mask to be applied, typically a binary array where 1 indicates 
                                a weight is kept and 0 indicates it is pruned.
            ret (int, optional): A flag to determine the type of return value:
                                - If 0, returns the computed objective function value (default).
                                - If 1, returns a tuple containing sparsity and accuracy of the model 
                                after applying the pruning solution.

        Returns:
            float or tuple: Depending on the `ret` value, returns either the objective function value 
                            or a tuple (sparsity, accuracy) reflecting the performance metrics of 
                            the pruned model.
        """
        return objective_function(self.model, self.layer_name, solution, self.val_loader, self.accuracy_lower_limit, ret)
    

    def performance_wrapper(self, solution, ret=0):
        """
        Wrapper method for the objective_function to evaluate the efficacy of a pruning solution based 
        on model sparsity and accuracy using the test dataset (instead of validation dataset). This method 
        allows for easy integration of the objective function within the genetic algorithm workflow by automatically 
        passing the relevant model, layer, and data loader.

        Parameters:
            solution (np.array): The pruning mask to be applied, typically a binary array where 1 indicates 
                                a weight is kept and 0 indicates it is pruned.
            ret (int, optional): A flag to determine the type of return value:
                                - If 0, returns the computed objective function value (default).
                                - If 1, returns a tuple containing sparsity and accuracy of the model 
                                after applying the pruning solution.

        Returns:
            float or tuple: Depending on the `ret` value, returns either the objective function value 
                            or a tuple (sparsity, accuracy) reflecting the performance metrics of 
                            the pruned model.
        """
        return objective_function(self.model, self.layer_name, solution, self.test_loader, self.accuracy_lower_limit, ret)
    

    def initial_population(self, pop_size, solution_size, initial_sparsity_ratio):
        """
        Generates an initial population of pruning masks for the genetic algorithm. Each individual in 
        the population represents a potential solution (pruning mask) for the model. This method initializes 
        the population with random binary arrays where the probability of each element being zero (pruned) 
        is determined by the initial sparsity ratio.

        Parameters:
            pop_size (int): The size of the population, i.e., the number of individual solutions to generate.
            solution_size (int): The size of each solution, which should match the number of parameters 
                                in the layer of the model that is being targeted for pruning.
            initial_sparsity_ratio (float): The proportion of weights to initially set as pruned (0) in each 
                                            solution. The rest will be set to keep (1).

        Returns:
            list of np.array: A list containing the initial population of solutions, where each solution 
                            is a numpy array of binary values (0s and 1s).
        """
        population = [np.random.choice([0, 1], size=(solution_size,), p=[initial_sparsity_ratio, 1-initial_sparsity_ratio]) for _ in range(pop_size)]
        return population
    

    def crossover(self, parent1, parent2, crossover_rate=0.9):
        """
        Performs the crossover operation in genetic algorithm to generate new offspring (solutions) from two parent solutions.
        This method uses a single-point crossover approach where a point on the parent solution arrays is chosen at random,
        and the tails beyond that point are swapped between the two parents to create two new children.

        Parameters:
            parent1 (np.array): The first parent solution array.
            parent2 (np.array): The second parent solution array.
            crossover_rate (float, optional): The probability of the crossover operation occurring between two parents.
                                            If a random draw falls below this rate, crossover happens; otherwise,
                                            the parents are returned without modification. Defaults to 0.9.

        Returns:
            tuple of np.array: A tuple containing two new solutions (children), each being a numpy array. These children are
                            either a result of the crossover (if it occurs) or direct copies of the original parents (if not).
        """
        if random.random() < crossover_rate:
            point = random.randint(1, len(parent1)-1)  # Crossover point
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        else:
            return parent1.copy(), parent2.copy()
        

    def mutation(self, solution, mutation_rate=0.01):
        """
        Applies mutation to a given solution, altering its elements to introduce variability and prevent premature convergence.
        This method iterates through each element of the solution array, and with a probability defined by the mutation rate,
        sets the element to 0 (representing a pruned weight). This stochastic process helps to explore new areas of the solution
        space that may not be reachable through crossover alone.

        Parameters:
            solution (np.array): The solution array to be mutated, typically a binary array where each element indicates
                                whether a corresponding weight is kept (1) or pruned (0).
            mutation_rate (float, optional): The probability of any single weight being mutated (set to 0). Defaults to 0.01.

        Returns:
            np.array: The mutated solution array.

        Mutation is a fundamental genetic operator that provides genetic diversity and enables the genetic algorithm to
        escape local optima by randomly altering solution elements. This method ensures that even well-performing solutions
        are subjected to random changes, thus simulating natural genetic variation and fostering robustness in the solution population.
        """
        for i in range(len(solution)):
            if random.random() < mutation_rate:
                solution[i] = 0
        return solution
    
    
    def load_population(self, population_size, solution_size, initial_sparsity_ratio):
        """
        Loads a population of pruning solutions from a file, or generates a new population if the file does not exist.
        This method attempts to retrieve a previously saved population specific to the model and layer from a pickle file.
        If the file cannot be found or an error occurs during the load process, it generates a new initial population
        using the specified sparsity ratio.

        Parameters:
            population_size (int): The number of individual solutions (population size) to be loaded or generated.
            solution_size (int): The number of parameters in the targeted layer, determining the size of each solution.
            initial_sparsity_ratio (float): The proportion of weights to initially set as pruned (0) in each solution,
                                            used if a new population needs to be generated.

        Returns:
            list of np.array: A population of solutions, where each solution is a numpy array of binary values (0s and 1s).
                            Each array represents a potential pruning mask for the neural network layer.
        """
        try:
            with open(f"./populations/{self.model_name}/population_{population_size}_{self.layer_name}.pkl", 'rb') as fp:
                population = pickle.load(fp)
        except:
            population = self.initial_population(population_size, solution_size, initial_sparsity_ratio)
        return population
    
    
    def genetic_algorithm(self, population_size, solution_size, crossover_rate, mutation_rate, generations, warm_start, initial_sparsity_ratio, sparsity_threshold):
        """
        Executes the genetic algorithm to optimize pruning masks for a neural network based on a defined objective function
        that evaluates sparsity and accuracy. The method handles initialization of the population, selection, crossover,
        mutation, and replacement over a number of generations.

        Parameters:
            population_size (int): The number of solutions in the population.
            solution_size (int): The number of parameters in the targeted layer, determining the size of each solution.
            crossover_rate (float): Probability with which two solutions will undergo crossover.
            mutation_rate (float): Probability with which any single element of a solution may be mutated.
            generations (int): Number of iterations the genetic algorithm should run.
            warm_start (bool): If True, the population is loaded from a previously saved state; otherwise, it is initialized anew.
            initial_sparsity_ratio (float): The proportion of weights initially set as pruned when creating a new population.
            sparsity_threshold (float): The sparsity level at which the algorithm will stop if achieved by any solution.

        Returns:
            tuple: A tuple containing the best score achieved and the corresponding best solution.

        This method advances through multiple generations, each time performing selection based on fitness, 
        breeding new solutions through crossover and mutation, and inserting them into the population using 
        a heap-based selection strategy to keep only the best solutions. It tracks and reports performance metrics 
        across generations, including average and best scores, and validation and test accuracies.
        The method also supports saving the state of the population and the best solution per generation if 
        required by the implementation settings, which allows for resuming the process or auditing the results later.
        """
        global args

        if warm_start:
            population = self.load_population(population_size, solution_size, initial_sparsity_ratio)
        else:
            population = self.initial_population(population_size, solution_size, initial_sparsity_ratio)
        population_heap = [(self.objective_function_wrapper(sol), idx, sol) for idx, sol in enumerate(population)]
        available_indices = set([len(population_heap), len(population_heap)+1])
        heapify(population_heap)
        for gen in range(generations):
            for i in range(len(population_heap)):
                _, _, x = random.choice(population_heap)
                _, _, y = random.choice(population_heap)
                c1, c2 = self.crossover(x, y, crossover_rate)
                c1 = self.mutation(c1, mutation_rate)
                c2 = self.mutation(c2, mutation_rate)
                idx1 = available_indices.pop()
                idx2 = available_indices.pop()
                heappush(population_heap, (self.objective_function_wrapper(c1), idx1, c1))
                heappush(population_heap, (self.objective_function_wrapper(c2), idx2, c2))
                _, idx1, _ = heappop(population_heap)
                _, idx2, _ = heappop(population_heap)
                available_indices.add(idx1)
                available_indices.add(idx2)

            best_score, _, best_sol = max(population_heap, key= lambda x : x[0])
            val_perf = self.objective_function_wrapper(best_sol, ret=1)
            test_perf = self.performance_wrapper(best_sol, ret=1)
            avg_score = sum(val for val, _, _ in population_heap)/len(population_heap)
            print(f"Generation {gen + 1}: Best Score = {best_score:.4f} | Best Sparsity {test_perf[0]:.4f} | Val Accuracy: {val_perf[1]:.4f} | Test Accuracy: {test_perf[1]:.4f} | Avg Score = {avg_score:.4f}")

            population = [sol for _, _, sol in population_heap]
            if args.save_results:
                with open(f"./populations/{self.model_name}/population_{population_size}_{self.layer_name}.pkl", 'wb') as fp:
                    pickle.dump(population, fp)

            if args.save_results:
                with open(f"./solutions/{self.model_name}/best_solution_{self.layer_name}.pkl", 'wb') as fp:
                    pickle.dump(best_sol, fp)

            if val_perf[0] >= sparsity_threshold:
                break

        best_score, _, best_sol = max(population_heap, key= lambda x : x[0])
        return best_score, best_sol

    
    def prune_one_layer(self, layer_name, config):
        """
        Prunes a specified layer of a neural network using the genetic algorithm configured through `config`. This method 
        orchestrates the pruning process by determining the best pruning solution for the layer, applying this solution, 
        retraining the model on the pruned layer, and finally removing the pruned weights permanently.

        Parameters:
            layer_name (str): The name of the layer to prune.
            config (dict): Configuration parameters for the genetic algorithm including population size, solution size,
                        crossover rate, mutation rate, number of generations, and other relevant settings.

        Returns:
            None. Outputs the best solution's score and performance directly and updates the model in-place.

        This method integrates several steps:
        - It runs the genetic algorithm to find the optimal pruning solution for the specified layer.
        - Applies the best pruning solution to the layer.
        - Retrains the model for a specified number of epochs to adjust to the changes introduced by pruning.
        - Removes the pruned weights permanently from the layer.
        - Evaluates and prints the test accuracy after pruning.
        - Checks and logs the sparsity achieved in the pruned layer.
        """
        global solution_mask

        self.layer_name = layer_name
        layer = get_module_by_name(self.model, self.layer_name)
        
        best_score, best_solution = self.genetic_algorithm(**config)
        print("Best Solution Score:", best_score)
        print("Best Solution Perf:", self.objective_function_wrapper(best_solution, ret=1))

        self.layer_solutions[self.layer_name] = best_solution

        solution_mask = torch.tensor(best_solution)
        layer = get_module_by_name(self.model, self.layer_name)
        custom_unstructured(layer, name='weight')

        train(self.model, self.train_loader, self.val_loader, self.optimizer, self.criterion, self.post_prune_epochs_per_layer, self.device)

        layer = get_module_by_name(self.model, self.layer_name)
        prune.remove(layer, 'weight')

        print("Test accuracy: ", get_accuracy(self.model, self.test_loader, self.device))

        self.check_sparsity()

    
    def prune_all_layers(self, config):
        """
        Prunes all the layers of a neural network that have trainable weights using a genetic algorithm. This method
        sequentially processes each layer, applies the genetic algorithm to find the optimal pruning solution, retrains
        the model to adapt to the pruned layer, and iteratively updates the entire model's structure.

        Parameters:
            config (dict): Configuration parameters for the genetic algorithm which include settings such as population size,
                        crossover rate, mutation rate, and other necessary parameters to execute the genetic pruning.

        Returns:
            None. This method updates the model in-place and prints out the performance metrics after pruning each layer and
            the entire model.

        The method goes through each layer, checks if it has trainable weights, and if so, proceeds to prune it based on the
        genetic algorithm. After pruning each layer, the model is retrained to ensure it adapts well to the changes. The final
        accuracy of the model on the test set is printed, and if configured, the pruned model's state is saved.

        During the process, it also keeps track of the test accuracies after each layer's pruning and after final retraining,
        providing insights into the model's performance progression as layers are pruned and retrained.
        """
        global solution_mask, args

        test_accuracies = []
        for idx, layer_name in enumerate(self.module_names):
            self.layer_name = layer_name
            layer = get_module_by_name(self.model, self.layer_name)

            print(f"Layer number: {idx}")
            print(f"Layer name: {layer_name}")

            try:
                layer.weight
            except:
                print("No attribute weight!")
                continue

            print(f"Layer shape: {layer.weight.size()}")
            print(f"Layer size: {np.prod(layer.weight.size())}")
            
            config["solution_size"] = np.prod(layer.weight.size())
            best_score, best_solution = self.genetic_algorithm(**config)
            print("Best Solution Score:", best_score)
            print("Best Solution Perf:", self.objective_function_wrapper(best_solution, ret=1))

            self.layer_solutions[self.layer_name] = best_solution

            for prev_layer_name, prev_solution in self.layer_solutions.items():
                solution_mask = torch.tensor(prev_solution)
                prev_layer = get_module_by_name(self.model, prev_layer_name)
                custom_unstructured(prev_layer, name='weight')

            train(self.model, self.train_loader, self.val_loader, self.optimizer, self.criterion, self.post_prune_epochs_per_layer, self.device)

            for prev_layer_name, prev_solution in self.layer_solutions.items():
                prev_layer = get_module_by_name(self.model, prev_layer_name)
                prune.remove(prev_layer, 'weight')

            print("Test accuracy: ", get_accuracy(self.model, self.test_loader, self.device))

            test_accuracies.append(get_accuracy(self.model, self.test_loader, self.device))
            print(test_accuracies)
            self.check_sparsity()
            if args.save_results:
                torch.save(self.model.state_dict(), f'./sparse_models/{self.model_name}/sparse_weights.pth')

        for prev_layer_name, prev_solution in self.layer_solutions.items():
            solution_mask = torch.tensor(prev_solution)
            prev_layer = get_module_by_name(self.model, prev_layer_name)
            custom_unstructured(prev_layer, name='weight')

        train(self.model, self.train_loader, self.val_loader, self.optimizer, self.criterion, self.post_prune_epochs, self.device)

        for prev_layer_name, prev_solution in self.layer_solutions.items():
            prev_layer = get_module_by_name(self.model, prev_layer_name)
            prune.remove(prev_layer, 'weight')

        print("Test accuracy: ", get_accuracy(self.model, self.test_loader, self.device))

        test_accuracies.append(get_accuracy(self.model, self.test_loader, self.device))
        print(test_accuracies)
        self.check_sparsity()
        if args.save_results:
            torch.save(self.model.state_dict(), f'./sparse_models/{self.model_name}/sparse_weights_retrained.pth')


    def prune_all_layers_iteratively(self, iters, sparsity_thresholds, config):
        """
        Iteratively prunes all trainable layers of a neural network multiple times across specified iterations, 
        adjusting the sparsity threshold each iteration based on provided thresholds. This method allows for 
        progressively deeper pruning with the ability to fine-tune the model's response to increased sparsity after 
        each iteration.

        Parameters:
            iters (int): Number of iterations to repeat the entire pruning process.
            sparsity_thresholds (list): List of sparsity thresholds for each iteration; controls how aggressive 
                                        the pruning should be in each iteration.
            config (dict): Configuration parameters for the genetic algorithm, including settings such as population 
                        size, crossover rate, mutation rate, and other necessary parameters to execute the genetic pruning.

        Returns:
            None. This method updates the model in-place and prints out performance metrics after pruning each layer 
            and the entire model in each iteration.

        During each iteration, this method goes through each layer of the model, applies genetic algorithm-based pruning, 
        retrains the model to adapt to the new sparsity level, and evaluates the model's performance. This approach 
        helps in achieving a desired global sparsity level while maintaining model performance as much as possible.
        After each iteration, the method optionally saves the state of the model, facilitating further analysis or deployment.
        """
        global solution_mask, args
        
        for iter in range(iters):
            test_accuracies = []
            config['sparsity_threshold'] = sparsity_thresholds[iter]
            for idx, layer_name in enumerate(self.module_names):
                self.layer_name = layer_name
                layer = get_module_by_name(self.model, self.layer_name)

                print(f"Layer number: {idx}")
                print(f"Layer name: {layer_name}")

                try:
                    layer.weight
                except:
                    print("No attribute weight!")
                    continue

                print(f"Layer shape: {layer.weight.size()}")
                print(f"Layer size: {np.prod(layer.weight.size())}")
                
                config["solution_size"] = np.prod(layer.weight.size())
                best_score, best_solution = self.genetic_algorithm(**config)
                print("Best Solution Score:", best_score)
                print("Best Solution Perf:", self.objective_function_wrapper(best_solution, ret=1))

                self.layer_solutions[self.layer_name] = best_solution

                for prev_layer_name, prev_solution in self.layer_solutions.items():
                    solution_mask = torch.tensor(prev_solution)
                    prev_layer = get_module_by_name(self.model, prev_layer_name)
                    custom_unstructured(prev_layer, name='weight')

                train(self.model, self.train_loader, self.val_loader, self.optimizer, self.criterion, self.post_prune_epochs_per_layer, self.device)

                for prev_layer_name, prev_solution in self.layer_solutions.items():
                    prev_layer = get_module_by_name(self.model, prev_layer_name)
                    prune.remove(prev_layer, 'weight')

                print("Test accuracy: ", get_accuracy(self.model, self.test_loader, self.device))

                test_accuracies.append(get_accuracy(self.model, self.test_loader, self.device))
                print(test_accuracies)
                self.check_sparsity()
                if args.save_results:
                    torch.save(self.model.state_dict(), f'./sparse_models/{self.model_name}/sparse_weights.pth')

            for prev_layer_name, prev_solution in self.layer_solutions.items():
                solution_mask = torch.tensor(prev_solution)
                prev_layer = get_module_by_name(self.model, prev_layer_name)
                custom_unstructured(prev_layer, name='weight')

            train(self.model, self.train_loader, self.val_loader, self.optimizer, self.criterion, self.post_prune_epochs, self.device)

            for prev_layer_name, prev_solution in self.layer_solutions.items():
                prev_layer = get_module_by_name(self.model, prev_layer_name)
                prune.remove(prev_layer, 'weight')

            print("Test accuracy: ", get_accuracy(self.model, self.test_loader, self.device))

            
            self.check_sparsity()
            if args.save_results:
                torch.save(self.model.state_dict(), f'./sparse_models/{self.model_name}/sparse_weights_retrained.pth')

            config['warm_start'] = True


    def check_sparsity(self):
        """
        Calculates and prints the sparsity levels for each layer in the model that has been pruned using the 
        solutions stored in `layer_solutions`. This method provides a measure of how many weights in each layer 
        have been set to zero (pruned) as a proportion of the total number of weights in that layer.

        Returns:
            None. Outputs the sparsity levels directly to the console.

        This method iterates through each pruned layer, retrieves the current weights from the model, converts them 
        to a NumPy array, and calculates the proportion of weights that are zero. The sparsity level for each layer 
        is then printed, giving an overview of the model's overall reduction in parameter count due to pruning.
        """
        sparsity_levels = []
        for layer_name in self.layer_solutions:
            layer = get_module_by_name(self.model, layer_name)
            layer_weights = copy.deepcopy(layer.weight)
            layer_weights = layer_weights.to('cpu')
            layer_weights = np.array(layer_weights.detach())
            sparsity = (layer_weights.size - np.count_nonzero(layer_weights))/layer_weights.size
            sparsity_levels.append(sparsity)

        print("Sparsity Levels: ", sparsity_levels)


def prune_resnet(layer_name):
    """
    Initiates the pruning process for the ResNet-18 model using genetic algorithm. This function sets up the model,
    loads pre-trained weights, and applies either layer-specific or whole-model pruning based on the specified
    `layer_name`.

    Parameters:
        layer_name (str or None): The name of a specific layer to prune. If None, the function will iteratively
                                  prune all layers of the model.

    Returns:
        None. The function directly modifies the model and optionally prints model accuracy and sparsity details.

    The function loads a ResNet18 model, its pre-trained weights, and dataset loaders for CIFAR-10. It prints the
    baseline accuracy of the model, initializes pruning settings, and then either prunes a specific layer or all
    layers based on the `layer_name` parameter.

    Pruning settings include the genetic algorithm's parameters such as population size, crossover rate, mutation
    rate, number of generations, and sparsity targets. The method supports both targeted layer pruning and
    comprehensive model pruning across multiple iterations with increasing sparsity levels.

    This approach allows for flexible and targeted pruning strategies, making it suitable for experiments on
    model efficiency and performance under different pruning scenarios.
    """

    net = ResNet18()
    device = 'cuda'
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    checkpoint = torch.load('./ResNet/resnet_ckpt.pth')
    net.load_state_dict(checkpoint['net'])

    train_loader, val_loader, test_loader = load_data_cifar10()

    print(f"Baseline accuracy: {get_accuracy(net, test_loader, device)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=5e-4)

    GP = GeneticPruning(net, train_loader, val_loader, test_loader, optimizer, criterion, device, \
                        post_prune_epochs=20, post_prune_epochs_per_layer=5, accuracy_lower_limit=0.88)

    if layer_name is not None:
        layer = get_module_by_name(net, layer_name)
        print(f"Layer name: {layer_name}")
        print(f"Layer shape: {layer.weight.size()}")
        print(f"Layer size: {np.prod(layer.weight.size())}")
        config = {
                    "population_size": 10,
                    "solution_size": np.prod(layer.weight.size()),
                    "crossover_rate": 1,
                    "mutation_rate": 0.05,
                    "generations": 20,
                    "warm_start": False,
                    "initial_sparsity_ratio": 0.1,
                    "sparsity_threshold": 1
        }
        GP.prune_one_layer(layer_name, config)
    else:
        print("Modules: ", GP.module_names)
        config = {
                    "population_size": 10,
                    "crossover_rate": 1,
                    "mutation_rate": 0.1,
                    "generations": 15,
                    "warm_start": False,
                    "initial_sparsity_ratio": 0.1,
                    "sparsity_threshold": 1
        }
        GP.prune_all_layers_iteratively(iters=2, sparsity_thresholds=[0.6, 1], config=config)


def prune_lenet(layer_name):
    """
    Initiates the pruning process for the LeNet-5 model using genetic algorithm. This function sets up the model,
    loads pre-trained weights, and applies either layer-specific or whole-model pruning based on the specified
    `layer_name`.

    Parameters:
        layer_name (str or None): The name of a specific layer to prune. If None, the function will prune
                                  all layers of the model.

    Returns:
        None. The function directly modifies the model and optionally prints model accuracy and sparsity details.

    The function loads a LeNet model, its pre-trained weights, and dataset loaders for MNIST. It prints the
    baseline accuracy of the model, initializes pruning settings, and then either prunes a specific layer or all
    layers based on the `layer_name` parameter.

    Pruning settings include the genetic algorithm's parameters such as population size, crossover rate, mutation
    rate, number of generations, and sparsity targets. The method supports both targeted layer pruning and
    comprehensive model pruning.
    """

    net = LeNet()
    device = 'cuda'
    net = net.to(device)
    checkpoint = torch.load('./LeNet/lenet_ckpt.pth')
    net.load_state_dict(checkpoint)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    train_loader, val_loader, test_loader = load_data_mnist()

    print(f"Baseline accuracy: {get_accuracy(net, test_loader, device)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=net.parameters(), lr=0.001)

    GP = GeneticPruning(net, train_loader, val_loader, test_loader, optimizer, criterion, device, \
                        post_prune_epochs=10, post_prune_epochs_per_layer=3, accuracy_lower_limit=0.6)

    if layer_name is not None:
        layer = get_module_by_name(net, layer_name)
        print(f"Layer name: {layer_name}")
        print(f"Layer shape: {layer.weight.size()}")
        print(f"Layer size: {np.prod(layer.weight.size())}")
        config = {
                "population_size": 10,
                "solution_size": np.prod(layer.weight.size()),
                "crossover_rate": 1,
                "mutation_rate": 0.05,
                "generations": 20,
                "warm_start": False,
                "initial_sparsity_ratio": 0.1,
                "sparsity_threshold": 1
        }
        GP.prune_one_layer(layer_name, config)
    else:
        print("Modules: ", GP.module_names)
        config = {
                    "population_size": 10,
                    "crossover_rate": 1,
                    "mutation_rate": 0.1,
                    "generations": 25,
                    "warm_start": False,
                    "initial_sparsity_ratio": 0.1,
                    "sparsity_threshold": 1
        }
        GP.prune_all_layers(config)


def fine_tune_sparse_model():
    """
    Fine-tunes a sparsely pruned model by loading its weights, reapplying the best pruning solutions,
    and training it further. This function aims to recover or improve the performance of the model after
    the initial pruning stages.

    The function follows these steps:
    1. Load the sparsely pruned model weights.
    2. Reapply the pruning masks based on the best solutions saved during the initial pruning.
    3. Conduct further training to fine-tune the model's performance.
    4. Optionally save the retrained model's weights for future use.

    Returns:
        None. The function prints out the model's accuracy before and after fine-tuning and updates
        the model weights in-place.

    This process helps in enhancing the model's accuracy that might have been degraded due to pruning
    and ensures that the model remains useful for its intended tasks.
    This function is particularly useful in scenarios where deep neural networks have undergone significant
    sparsity increases, requiring adjustments to their learned parameters to maintain or improve performance.
    """
    global solution_mask, args

    net = ResNet18()
    device = 'cuda'
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    model_name = net.module.__class__.__name__
    checkpoint = torch.load(f'./sparse_models/{model_name}/sparse_weights_retrained.pth')
    net.load_state_dict(checkpoint)

    train_loader, val_loader, test_loader = load_data_cifar10()

    module_names = []
    get_module_names(net, '', module_names)

    layer_solutions = {}

    for layer_name in module_names:
        try:
            with open(f"./solutions/{model_name}/best_solution_{layer_name}.pkl", 'rb') as fp:
                best_solution = pickle.load(fp)
        except:
            continue
        layer_solutions[layer_name] = best_solution
        print(f"Layer {layer_name} masked!")


    for layer_name, solution in layer_solutions.items():
        solution_mask = torch.tensor(solution)
        layer = get_module_by_name(net, layer_name)
        custom_unstructured(layer, name='weight')

    print("Test accuracy: ", get_accuracy(net, test_loader, device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=5e-4)
    train(net, train_loader, val_loader, optimizer, criterion, 20, device=device)

    for layer_name, solution in layer_solutions.items():
        layer = get_module_by_name(net, layer_name)
        prune.remove(layer, 'weight')

    print("Test accuracy: ", get_accuracy(net, test_loader, device))

    if args.save_results:
        torch.save(net.state_dict(), f'./sparse_models/{model_name}/sparse_weights_retrained.pth')


def main():
    """
    Main entry point for running the dynamic pruning and fine-tuning process using a genetic algorithm.
    This function parses command-line arguments to determine the model type, the specific layer to prune,
    and whether to fine-tune or just prune the model.

    The function supports pruning and fine-tuning for both ResNet-18 and LeNet-5 models. It allows for
    specific layer pruning or whole-model pruning based on the provided arguments. It also includes an
    option to save results to disk, which is useful for persistent storage or later analysis.

    Command-line Arguments:
        --resnet: Flag to select the ResNet-18 model for pruning.
        --lenet: Flag to select the LeNet-5 model for pruning.
        --layer: Specifies the name of the layer to prune. If not provided, all layers are pruned.
        --finetune: Flag to fine-tune the model after pruning.
        --save_results: Flag to save the pruning and fine-tuning results to disk
                        (please do not specify if experimenting).

    Returns:
        None. Based on the arguments, the function executes the specified pruning and/or fine-tuning
        operations and prints relevant outputs.

    Example Command Line Usage:
        # To prune a specific layer of the ResNet-18 model
        python pruning.py --resnet --layer module.layer4.0.conv2

        # To prune all the layers of the LeNet-5 model
        python pruning.py --lenet

        # To fine-tune a previously pruned model 
        (currently only ResNet-18 is supported, LeNet-5 doesn't require further fine-tuning)
        python pruning.py --finetune

    This flexible command-line interface allows users to easily switch between different models and pruning
    strategies without altering the codebase, making it suitable for automated workflows and experimentation.
    """
    global args

    parser = argparse.ArgumentParser(description='dynamic pruning using genetic algorithm')
    parser.add_argument('--resnet', action='store_true', help='prune ResNet-18 model')
    parser.add_argument('--lenet', action='store_true', help='prune LeNet-5 model')
    parser.add_argument('--layer', default=None, type=str, help='name of the layer to prune (do not specify to prune all layers)')
    parser.add_argument('--finetune', action='store_true', help='finetune pruned model')
    parser.add_argument('--save_results', action='store_true', help='store results on disk (do not specify if experimenting)')
    args = parser.parse_args()

    if args.resnet:
        prune_resnet(args.layer)
    elif args.lenet:
        prune_lenet(args.layer)
    elif args.finetune:
        fine_tune_sparse_model()
    

if __name__ == "__main__" :
    main()
