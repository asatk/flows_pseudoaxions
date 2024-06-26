"""
Author: Anthony Atkinson
Modified: 2023.07.15

I/O functions that verify or save to paths in the LFS (local file system).
"""


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
