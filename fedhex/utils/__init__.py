"""
Author: Anthony Atkinson
Modified: 2023.07.15

Utility functions subpackage
"""


from ._logging import get_loglevel, print_msg, set_loglevel, LOG_DEBUG, \
    LOG_INFO, LOG_WARN, LOG_ERROR, LOG_FATAL, MSG_BOLD, MSG_QUIET, MSG_LONG


__all__ = [
    "get_loglevel",
    "print_msg",
    "set_loglevel",
    "LOG_DEBUG",
    "LOG_INFO",
    "LOG_WARN",
    "LOG_ERROR",
    "LOG_FATAL",
    "MSG_BOLD",
    "MSG_QUIET",
    "MSG_LONG",
]