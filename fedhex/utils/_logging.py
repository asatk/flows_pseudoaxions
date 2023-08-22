"""
Author: Anthony Atkinson
Modified: 2023.07.20

Logging
"""


from datetime import datetime, timedelta
from sys import stdout
from textwrap import fill as twfill
from traceback import print_stack

#TODO perhaps allow setting default msg type as well
#TODO perhaps allow different time formats

loglevel: int= 1

LOG_INFO = 0
LOG_DEBUG = 1
LOG_WARN = 2
LOG_ERROR = 3
LOG_FATAL = 4

MSG_BOLD = 0
MSG_QUIET = 1
MSG_LONG = 2


level_codes: list[str] = [
    "I",
    "D",
    "W",
    "E",
    "F"
]

msgs: list[str] = [
    # A bolder message (wrapping)
    (
        "========================================"
        "========================================"
        "\n{:s}\n"
        "========================================"
        "========================================"
    ),

    # A quieter message (wrapping)
    (
        "\n{:s}\n"
    ),
    
    # A longer message (no wrapping)
    (
        "\n{:s}\n"
    )
]


def set_loglevel(level: int) -> None:
    """
    Set the global minimum logging level.

    level : int
        The global minimum importance of a log message for it to be printed.
    """
    global loglevel
    loglevel = level


def get_loglevel() -> int:
    """
    Returns the current global minimum logging level.

    Returns
        The global minimum importance of a log message for it to be printed.
    """
    return loglevel


def print_msg(msg: str, level: int=LOG_INFO, m_type: int=MSG_BOLD, time: datetime=None) -> datetime:
    """
    Utility method for logging output. Returns a datetime object indicating
    the time of logging if no time was provided. Otherwise, the time argument
    is returned to the call. If the global logging level is higher than that
    provided, the message is not printed but a time is still returned.

    msg : str
        the message to be printed.
    level : int (default=LOG_INFO)
        the importance of the logged message. Messages with a level of
        `LOG_FATAL` will always print the stacktrace and exit.
    m_type : int (default=MSG_BOLD)
        the format of the logged message.
    time : datetime (default=None)
        the relevant time for the logged message. If no time is provided, the
        current time as given by `datetime.now()` will be used as the log time.
        The time will always be printed in the format "%H:%M:%S.%2f"

    Returns
        The time relevant to the log message. If `time` is None, the current
        time is returned.
    """

    # TODO add an open filestream to have output to a log file
    # TODO nicely print stack (not the terrible python printout)
    # TODO perhaps specify time str-conversion format as argument?
    if time is None:
        time = datetime.now()
    
    if isinstance(time, datetime):
        timestr = time.strftime("%H:%M:%S.%f")[:-4]
    elif isinstance(time, timedelta):
        timestr = str(time)[-15:-4]
    else:
        # TODO figure something out here later
        print("Bad time format. Must be one of [`datetime`, `timedelta`]")

    if level >= loglevel:
            
        if m_type in [MSG_BOLD, MSG_QUIET]:
            s = twfill("{:<s} <{:>s}> {:>s}".format(timestr,
                level_codes[level], msg), width=80)
        elif m_type is MSG_LONG:
            s = "{:s}\t<{:s}> {:s}".format(timestr, level_codes[level], msg)

        print(msgs[m_type].format(s))

    if level == LOG_FATAL:
        print_stack(file=stdout)
        exit()
    
    return time
