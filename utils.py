'''Utility functions required for the main pruning logic.'''

import os
import sys
import time

from functools import reduce
from typing import Union

import torch
import torch.nn as nn
import torch.nn.init as init


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    """
    Displays and updates a console progress bar with custom messages.

    Parameters:
        current (int): Current progress count. This should be an integer specifying the current step.
        total (int): Total count. This is the total number of steps as an integer.
        msg (str, optional): Additional information to display on the bar. This should be a string containing
                             any additional data or status you want to show on the bar.

    The function will display a progress bar in the format of:
    [=======>............]  Step: 0:00:01 | Tot: 0:00:10 | Message

    The bar shows the current progress, the time taken for the current step, the total time since the start,
    and an additional custom message if provided.

    Note:
        This function assumes access to `sys.stdout` and will not work as expected in environments where `sys.stdout`
        is redirected or unavailable, such as some IDEs or non-interactive shells.
    """
    global last_time, begin_time

    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    """
    Converts a time duration from seconds to a formatted string showing days, hours, minutes, 
    seconds, and milliseconds.

    The function will format the time to include up to two significant units. For example, it 
    will show days and hours, or hours and minutes, depending on the total duration. If the time 
    is less than a second, it will show milliseconds.

    Parameters:
        seconds (float): The total time duration in seconds.

    Returns:
        str: A string representing the time duration formatted in human-readable form. The format 
             can include days (D), hours (h), minutes (m), seconds (s), and milliseconds (ms). It 
             prioritizes higher units over lower units and limits the output to two significant units.
    """

    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_accuracy(net, test_loader, device):
    """
    Computes the accuracy of a neural network model on a given dataset.

    Parameters:
        net (torch.nn.Module): The neural network model to be evaluated.
        test_loader (torch.utils.data.DataLoader): The DataLoader that provides the dataset over which the 
                                                   model's accuracy is to be calculated. It should yield batches
                                                   of input data and corresponding labels.
        device (str or torch.device): The device (e.g., 'cpu' or 'cuda') on which the model and data should be 
                                      loaded for evaluation.

    Returns:
        float: The accuracy of the model, calculated as the ratio of correctly predicted observations to the 
               total observations in the provided dataset.
    """
    
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return correct/total


def get_module_by_name(module: Union[torch.Tensor, nn.Module],
                       access_string: str):
    """
    Retrieves a sub-module from a PyTorch module hierarchy using a dot-separated access string.

    Parameters:
        module (Union[torch.Tensor, nn.Module]): The root module from which to retrieve the sub-module.
        access_string (str): A string that specifies the path to the sub-module using dot notation. 
                             For example, 'layer1.0.conv1' refers to the 'conv1' module within the 
                             first sub-module of 'layer1'.

    Returns:
        nn.Module: The sub-module at the specified path within the module hierarchy. If the path is 
                   incorrect or the specified module does not exist, it raises an AttributeError.

    This function facilitates dynamic access to any part of a complex model architecture without 
    hard-coding the access to sub-modules, thus enhancing the flexibility and maintainability of 
    code that needs to interact with specific parts of a model.
    """

    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def get_module_names(module, parent_name, module_list):
    """
    Recursively collects the names of all modules within a given PyTorch module hierarchy.

    Parameters:
        module (nn.Module): The root module from which to start collecting module names.
        parent_name (str): The dot-separated prefix representing the hierarchical path to the current module.
                           This should be an empty string for the root module.
        module_list (list): A list that accumulates the names of all modules. This should initially be an empty list 
                            which will be populated by the function.

    This function traverses the module hierarchy starting from the given module and appends each module's 
    name to the module_list, formatted as a dot-separated string that represents its path within the model 
    (e.g., 'module.submodule.subsubmodule').

    Example usage:
        model = torchvision.models.resnet18(pretrained=True)
        modules = []
        get_module_names(model, '', modules)
        print(modules)  # Output will include names like 'conv1', 'layer1.0.conv1', etc.

    This function is particularly useful for dynamic inspection of models, allowing programmers to list or access 
    components of a model without prior knowledge of its architecture. This can be crucial for tasks such as 
    automated model modification, visualization, or selective parameter freezing / masking.
    """

    module_name = parent_name
    children = [child for child in module.named_children()]
    if children:
        for name, child in children:
            child_name = module_name + ('.' if module_name else '') + name
            get_module_names(child, child_name, module_list)
    else:
        module_list.append(module_name)
