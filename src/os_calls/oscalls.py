import os
from os import system

import torch


def is_running_in_docker():
    """
    checks if its running docker path may differ between docker and non docker
    :return:
    """
    # Check for the presence of the Docker environment file
    if os.path.exists('/.dockerenv'):
        return True

    # Check for the presence of Docker-specific cgroup files
    try:
        with open('/proc/1/cgroup', 'rt') as f:
            return 'docker' in f.read()
    except FileNotFoundError:
        return False


def check_cuda_available():
    """
    check if the cuda is available
    Returns
    -------

    """
    if torch.cuda.is_available():
        return True
    else:
        return False


class OsCalls:
    def __init__(self):
        import platform
        # Get the operating system name
        self.os_name = platform.system()

    def clear(self):
        """
        clears the screen windows or linux
        :return: no return
        """
        # for windows
        if self.os_name == 'Windows':
            _ = system('cls')
        # for mac and linux(here, os.name is 'posix')
        else:
            _ = system('clear')
