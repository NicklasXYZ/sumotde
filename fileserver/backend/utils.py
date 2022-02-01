from typing import Union, Any, Dict, List
from functools import wraps
import subprocess
import time
import re
import logging
import os
import random
import string

import numpy as np
import pandas as pd
from lxml import etree


ansi_escape = re.compile(r'''
    \x1B  # ESC
    (?:   # 7-bit C1 Fe (except CSI)
        [@-Z\\-_]
    |     # or [ for CSI, followed by a control sequence
        \[
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    )
''', re.VERBOSE)


def sec2str_1(sec):
    """ Convert seconds to the time format "HH:MM:SS".

    Args:
        sec (str, int or float): Seconds.

    Returns:
        (str): A time with format "HH:MM::S" (e.g. "13:30:03").
    """
    if is_number(sec):
        sec = float(sec)
    return time.strftime("%H:%M:%S", time.gmtime(sec))


def sec2str_2(sec):
    """ Convert seconds to the time format "HH.MM" (e.g. "13.50").

    Args:
        sec (str, int or float): Seconds.

    Returns:
        (str): A time with format "HH:MM" (e.g. "13:30").
    """
    if is_number(sec):
        sec = float(sec)
    return time.strftime("%H.%M", time.gmtime(sec))


def run_subprocess(commandline_args: str) -> None:
    counter = 0; _exit = False; error_msg = ""
    while True:
        logging.info(
            f"CMD args: {commandline_args}"
        )
        s = subprocess.Popen(
            commandline_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
        )
        (stdout_data, stderr_data) = s.communicate()
        # Check the returncode to see whether the process terminated normally
        if s.returncode == 0:
            logging.info(
                f"Subprocess exited normally with return code: {str(s.returncode)} \
                running command: {commandline_args}"
            )
            break
        elif counter >= 5:
            error_msg = f"Subprocess exited with non-zero return code: \
                {str(s.returncode)} multiple times: {str(counter)} running \
                command: {commandline_args}"
            logging.error(error_msg)
            _exit = True
        else:
            print("\nSUBPROCESS STDOUT DATA: ")
            print(stdout_data)
            print()
            print("\nSUBPROCESS STDERR DATA: ")
            print(stderr_data)
            error_msg = f"Subprocess exited with non-zero return code: \
                {str(s.returncode)} running command: {commandline_args}"
            logging.error(error_msg)
            _exit = True
        counter += 1
        if _exit is True:
            raise SystemExit(error_msg)


def is_number(s):
    """ Simply check whether or not a given input is a string or a number

    Args:
        s (str): A string.

    Returns:
        (bool): True if the string can be cast as a float (it is actually a
            number). False if the string can not be cast as a float (it is 
            actually not a number).
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_list(l):
    """ Check whether or not a given input is a list.

    Args:
        l (*): Data of any type.

    Returns:
        (bool): True if the given data is of type list. False otherwise.
    """
    if type(l) is list:
        return True
    else:
        return False


def timing(f):
    """Define a decorator function to time a function call.

    Note:
        The function f is called in an ordinary manner and the result is 
        returned. The time of the function call is simply printed to stdout.

    Args:
        f (function): The function to be timed.

    Returns:
        wrapper (function): The result of the function f.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        start_sysusr = time.process_time()
        start_wall = time.time()
        result = f(*args, **kwargs)
        end_sysusr = time.process_time()
        end_wall = time.time()
        print(
            "Call to "
            + str(f)
            + ". Elapsed time (sysusr): {}".format(end_sysusr - start_sysusr)
        )
        print(
            "Call to "
            + str(f)
            + ". Elapsed time (wall)  : {}".format(end_wall - start_wall)
        )
        return result
    return wrapper


def read_xml_file(
    xml_file: str,
    max_wait: float = 60.0,
    dtd_validation: bool = False
    ):
    start_time = time.time()
    while not os.path.exists(xml_file):
        logging.info(
            f"Waiting for file {xml_file}. Exists: {os.path.exists(xml_file)}"
        )
        if time.time() - start_time >= max_wait:
            logging.error(f"The file {xml_file} does not exist.")
            break
        time.sleep(1.0)
    logging.info(f"Reading .xml file: {xml_file}")
    parser = etree.XMLParser(dtd_validation=dtd_validation, no_network=True)
    root = etree.parse(xml_file, parser).getroot()
    return root


def file_path_error(path: str) -> None:
    logging.info("The given file path does not exist: ")
    logging.info(f"Relative file path                : {os.path.relpath(path)}")
    logging.info(f"Absolute file path                : {os.path.abspath(path)}")


def get_capacity(priority: int, maxspeed: float, lanes: int):
    road_class = -priority
    if (road_class == 0 or road_class == 1):
        return lanes * 2000.
    elif (road_class == 2 and maxspeed <= 11.):
        return lanes * 1333.33
    elif (road_class == 2 and maxspeed > 11. and maxspeed <= 16.):
        return lanes * 1500.
    elif (road_class == 2 and maxspeed > 16.):
        return lanes * 2000.
    elif (road_class == 3 and maxspeed <= 11.):
        return lanes * 800.
    elif (road_class == 3 and maxspeed > 11. and maxspeed <= 13.):
        return lanes * 875.
    elif (road_class == 3 and maxspeed > 13. and maxspeed <= 16.):
        return lanes * 1500.
    elif (road_class == 3 and maxspeed > 16.):
        return lanes * 1800.
    elif ((road_class >= 4 or road_class == -1) and maxspeed <= 5.):
        return lanes * 200.
    elif ((road_class >= 4 or road_class == -1) and maxspeed > 5. and maxspeed <= 7.):
        return lanes * 412.5
    elif ((road_class >= 4 or road_class == -1) and maxspeed > 7. and maxspeed <= 9.):
        return lanes * 600.
    elif ((road_class >= 4 or road_class == -1) and maxspeed > 9. and maxspeed <= 11.):
        return lanes * 800.
    elif ((road_class >= 4 or road_class == -1) and maxspeed > 11. and maxspeed <= 13.):
        return lanes * 1125.
    elif ((road_class >= 4 or road_class == -1) and maxspeed > 13. and maxspeed <= 16.):
        return lanes * 1583.
    elif ((road_class >= 4 or road_class == -1) and maxspeed > 16. and maxspeed <= 18.):
        return lanes * 1100.
    elif ((road_class >= 4 or road_class == -1) and maxspeed > 18. and maxspeed <= 22.):
        return lanes * 1200.
    elif ((road_class >= 4 or road_class == -1) and maxspeed > 22. and maxspeed <= 26.):
        return lanes * 1300.
    elif ((road_class >= 4 or road_class == -1) and maxspeed > 26.):
        return lanes * 1400.
    return lanes * 800.


def str_is_number(s: str) -> bool:
    """Simply check whether a given input is a string or a number.

    Args:
        s (str): A string.

    Returns:
        (bool): True if the string can be cast as a float (it is actually a number).
            False if the string can not be cast as a float (it is actually not a
            number).
    """
    try:
        float(s)
        return True
    except ValueError:
        return False


def str_is_bool(s: str) -> bool:
    """Simply check whether a given input is a boolean value.

    Args:
        s (str): A string.

    Returns:
        (bool): True if the string is a boolean value. False if the string
            is not.
    """
    if s.lower() in ("yes", "true", "t", "y", "1", "no", "false", "f", "n", "0"):
        return True
    else:
        return False


def str_to_bool(s: str) -> Union[None, bool]:
    """Convert a string to a boolean value.

    Args:
        s (str): A string.

    Returns:
        (bool): True if the string has boolean value "True". False if the string
            has boolean value "False".
    """
    if s.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        return None


def str_to_int(s: str) -> Union[None, int]:
    return int(float(s))


def get_random_string(length: int) -> str:
    """Generate a random string of letters.

    Args:
        length (int): The length of the string of random letters that should
        be generated.

    Returns:
        str: A random string of letters of a particular length.
    """
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(length))


# Generate random travel demand between different origin-destinations pairs
def generate_constant_demand_vector(
    scenario_data,
    value: float = 0.0,
    ) -> np.ndarray:
    n = scenario_data.network.total_od_pairs
    m = scenario_data.aggregation_intervals
    demand_vec = np.zeros(shape=n*m) + value
    return demand_vec


def generate_uniform_random_demand_vector(
    scenario_data,
    low: float,
    high: float,
    random_seed: Union[None, int] = None,
    ) -> np.ndarray:
    # Set the random seed for reproducibility purposes
    if random_seed is None:
        logging.info(
            "Using the 'random_seed' attribute set on the passed \
            'scenario_data' object as the seed for generating random numbers."
        )
        np.random.seed(seed=scenario_data.random_seed)
    else:
        logging.info(
            "Using the passed 'random_seed' value as the seed for generating \
            random numbers."
        )
        np.random.seed(seed=random_seed)
    n = scenario_data.network.total_od_pairs
    m = scenario_data.aggregation_intervals
    demand_vec = np.random.uniform(low=low, high=high, size=n*m)
    return demand_vec


def construct_x_vec(scenario_data, demand_vec: np.ndarray) -> pd.DataFrame:
    demands = []; total_demand = 0; counter = 0
    for i in range(0, scenario_data.aggregation_intervals):
        begin = scenario_data.interval_start + (
            scenario_data.interval_increment * i
        )
        end = scenario_data.interval_start + (
            scenario_data.interval_increment * (i + 1)
        )
        for j in range(0, scenario_data.network.taz["id"].shape[0]):
                if len(scenario_data.network.taz.iloc[j, :]["sources"]) != 0:
                    for k in range(0, scenario_data.network.taz["id"].shape[0]):
                        if len(scenario_data.network.taz.iloc[k, :]["sinks"]) != 0:
                            if j != k:
                                value = demand_vec[counter]
                                from_ = scenario_data.network.taz["id"][j]
                                to_ = scenario_data.network.taz["id"][k]
                                demand = {
                                    "value": value,
                                    "from": from_,
                                    "to": to_,
                                    "begin": begin,
                                    "end": end,
                                    "slice": i,
                                }
                                demands.append(demand)
                                total_demand += value; counter += 1
    logging.info(f"Total demand distributed: {str(total_demand)}")
    df = pd.DataFrame(data=demands, dtype=object)
    _types = {
        "value": float, 
        "from": str,
        "to": str,
        "begin": float,
        "end": float,
        "slice": int,
    }
    df = df.astype(dtype=_types)
    return df
