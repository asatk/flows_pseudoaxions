"""
Author: Anthony Atkinson
Modified: 2023.07.15

I/O functions that verify or save to paths in the LFS (local file system).
"""

# import argparse
import json
import os
import shutil
from typing import Callable


from ..utils import LOG_DEBUG, LOG_ERROR, LOG_FATAL, LOG_INFO, LOG_WARN, MSG_QUIET, print_msg


def init_service(new_files: list[str]=None, new_dirs: list[str]=None,
                 existing_files: list[str]=None, existing_dirs: list[str]=None,
                 overwrite: bool=False, mk_dir: bool=False,
                 init_fn: Callable[..., bool]=None, *args, **kwargs) -> bool:
    """
    Performs all necessary checks and path initialization for an arbitrary
    service.
    
    Any paths that are desired to be new files or directories will
    be checked if they are already present. If present and overwrite is False,
    the user will receive a prompt notifying them of the potential for
    overwriting. Otherwise, the paths will be forcibly overwritten.

    Any paths that are desired to already exist will be checked for their
    presence in the file system. If not present, the user will receive a prompt
    notifying them of the lack of file presence and if they wish to create a
    file/dir at that path. Otherwise, no path will be created and the init
    procedure will return `False`.

    Any additional initialization behavior can be passed in the argument
    `init_fn` with positional arguments in `args` and keyword arguments in
    `kwargs`.
    """


    for path in new_files:
        if os.path.isfile(path) and not _replace_prompt(path, overwrite, is_dir=False):
            return False
        
    for path in new_dirs:
        if os.path.isdir(path) and not _replace_prompt(path, overwrite, is_dir=True):
            return False

    for path in existing_files:
        if not os.path.isfile(path):
            print_msg(f"The file at '{path}' does not exist.", level=LOG_WARN)
            return False

    for path in existing_dirs:
        if not os.path.isdir(path):
            print_msg(f"The directory at '{path}' does not exist.",
                      level=LOG_WARN)
            if mk_dir:
                os.mkdir(path)
                print_msg(f"The directory at '{path}' has now been created.",
                      level=LOG_INFO)

            return False

    if init_fn is not None:
        init_success = init_fn(*args, **kwargs)
        if not init_success:
            print_msg("The provided initialization function did not succeed.",
                      level=LOG_WARN)
        return init_success

    return True


def _replace_prompt(path: str, replace: bool=False, is_dir: bool=False):
    """
    Prompt user with the option to replace an already-existing file. Returns
    True upon replacement, meaning the provided path can be written to. Returns
    False if the user does not want the data stored at the path to be replaced.

    path : str
        Path
    """

    if not os.path.isfile(path) and not os.path.isdir(path):
        print_msg(f"The provided path '{path}' is neither a " + \
                  "file nor directory.", level=LOG_WARN)
        return False
    
    prompt_str = f"Replace data stored at '{path}'? [Y/n]"
    n_str = f"Not removing data stored at '{path}'"
    y_str = f"Removing data stored at '{path}'"
    
    c = "Y" if replace else " "
    while c not in ["Y", "n"]:
        c = input(prompt_str)
        
    if c == "n":
        print_msg(n_str, level=LOG_WARN)
        return False
    else:
        print_msg(y_str, level=LOG_DEBUG, m_type=MSG_QUIET)
        
        if is_dir:
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            os.remove(path)

        return True


def save_config(path: str, params_dict: dict, save_all: bool=False) -> bool:

    """
    Save a configuration for a given service. The parameters passed to this
    function will be saved as a single dictionary in a .JSON file.

    Forcing the replacement of an already-existing config file is not possible
    with this function as it is highly recommended to keep track on each
    configuration without the possibility of overwriting them.
    

    path : str
        Where the config file is saved
    params_dict : dict
        Dictionary of parameters desired to be saved.
    save_all : bool (default=False)
        Allows all members of a dictionary to be saved. This includes hidden
        members that begin with '_'. This option is only useful if a user
        desires to save the contents of an entire module.

    Returns
        bool, True if the config file does not already exist or if it is
        requested to be replaced. False if the config file does already exist
        but the user does not desire to replace it.
    """

    if os.path.isfile(path) and not _replace_prompt(path, replace=False, is_dir=False):
        return False

    if save_all:
        config = params_dict
    else:
        config = dict([item for item in params_dict.items() if not item[0].startswith('_')])
    
    with open(path, "w") as config_file:
        json.dump(config, config_file, indent=4)

    return True


def load_config(path: str) -> dict|None:

    with open(path, "r") as json_file:
        config = json.load(json_file)

    if not isinstance(config, dict):
        print_msg(f"Config file at '{path}' is not of the correct format." + \
                  "These data must be saved as a single dictionary.",
                  level=LOG_ERROR)
        return None
    
    return config
    
