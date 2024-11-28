import os
import random

import numpy as np
import torch
import yaml


def get_absolute_path(file_name: str, bc_type: str = None) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file = file_name
    if bc_type is not None:
        file += '_' + bc_type
    file_name = os.path.join(script_dir, file)
    return file_name


def read_config_file(file_name):
    file_path = get_absolute_path(file_name)
    try:
        with open(file_path, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None


def np_load(file_name: str, bc_type: str = None):
    file_path = get_absolute_path(file_name, bc_type)
    try:
        data = np.load(file_path)
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None


def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return "%dh %02dm %02ds" % (hours, minutes, seconds)
    elif minutes > 0:
        return "%dm %02ds" % (minutes, seconds)
    else:
        return "%ds" % seconds


def format_memory_usage(memory_bytes):
    if memory_bytes < 1024:
        return f"{memory_bytes} B"
    elif 1024 <= memory_bytes < 1024 ** 2:
        return f"{memory_bytes / 1024:.3f} KB"
    elif 1024 ** 2 <= memory_bytes < 1024 ** 3:
        return f"{memory_bytes / (1024 ** 2):.3f} MB"
    elif 1024 ** 3 <= memory_bytes < 1024 ** 4:
        return f"{memory_bytes / (1024 ** 3):.3f} GB"
    else:
        return f"{memory_bytes / (1024 ** 4):.3f} TB"


def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_memory_usage():
    allocated_memory = torch.cuda.memory_allocated()
    max_allocated_memory = torch.cuda.max_memory_allocated()

    formatted_allocated_memory = format_memory_usage(allocated_memory)
    formatted_max_allocated_memory = format_memory_usage(max_allocated_memory)

    print(f"Memory allocated: {formatted_allocated_memory}, Max memory allocated: {formatted_max_allocated_memory}",
          flush=True)


def print_epoch_number_from_checkpoint(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    print(f"Epoch number: {checkpoint['epoch']}")


def print_epoch_for_all():
    print_epoch_number_from_checkpoint('model/lksd/checkpoint_DD.pth')
    print_epoch_number_from_checkpoint('model/lksd/checkpoint_DN.pth')
    print_epoch_number_from_checkpoint('model/lksd/checkpoint_ND.pth')
    print_epoch_number_from_checkpoint('model/lksd/checkpoint_NN.pth')