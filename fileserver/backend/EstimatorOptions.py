"""
This module contains a base-class for handling algorithm options and settings. \
    The classes handle default and user-defined algorithm options and resolves \
    any possible conflicts.
"""
from builtins import NotImplementedError
from abc import abstractmethod
from typing import Any, Dict

import logging
import numpy as np
from scipy.linalg import hadamard


class BaseOptions:

    def __init__(self, options: Dict[str, Any]) -> None:
        """Class constructor.

        Args:
            options (dict): All user defined algorithm options.
        """
        self.__options = self.set_options(options)
        logging.info("The following settings were set: ")
        for option in self.__options:
            logging.debug(
                "VALUE   : " + str(option) + ": " + str(self.__options[option])
            )

    def get_value(self, key: str) -> Any:
        """Return a certain value of an default or user-defined option
        by provifing the name of the option as a key.

        Args:
            key (str): The name of the option.

        Returns:
            Any: The corresponding value of the option.
        """
        try:
            return self.__options.get(key)
        except KeyError:
            print(
                "The requested key: {key} is not present in the dictionary!"
            )

    def get_options(self) -> Dict[str, Any]:
        """A getter method that returns all options that have been set.

        Returns:
            dict: All options that have been set.
        """
        return self.__options

    @abstractmethod
    def set_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        return options


class BaseEstimatorOptions(BaseOptions):
    """
    Base class for handling algorithm options.
    """

    def set_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        options = self.check_required_options(options)
        options = self.set_default_options(options)
        options = self.set_algorithm_options(options)
        return options

    def check_required_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        if "x0_size" not in options:
            raise KeyError(
                "The size of the input has not been specified!"
                + 'set: options = {"x0_size": x0.shape[0], ...} '
            )
        if "weight_configuration" not in options:
            raise KeyError(
                "The objective function weight configuration has not been \
                 specified!",
            )
        return options

    def set_default_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        ########################################################################
        # Set the random seed
        options["random_seed"] = int(options.get("random_seed", 1))
        ########################################################################
        # Maximum number of algorithm iterations
        max_iters = options["x0_size"]
        options["max_iters"] = int(options.get("max_iters", max_iters))
        ########################################################################
        # Maximum number of objective function evaluations (a soft upper bound)
        max_evals = max_iters * np.round(options["x0_size"] * 0.50)
        options["max_evals"] = int(options.get("max_evals", max_evals))
        ########################################################################
        # The stopping criteria to use
        convergence_criteria = options.get("stopping_criteria", "max_iters")
        if not isinstance(convergence_criteria, list):
            options["stopping_criteria"] = [convergence_criteria]
        ########################################################################
        # Set the absolute tolerance for the magnitude of the gradient
        options["ghat_magnitude_atol"] = float(
            options.get("ghat_magitude_atol", 1e-08)
        )
        # Set the absolute tolerance for stopping when the movement between
        # iterates becomes small
        options["x_movement_atol"] = float(
            options.get("x_movement_atol", 1e-08)
        )
        # Set the absolute tolerance for stopping when the movement between
        # objective function values become small
        options["of_saturation_atol"] = float(
            options.get("of_saturation_atol", 1e-08)
        )
        return options

    @abstractmethod  # Can be implemented by the user in an inheriting class
    def set_algorithm_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("TODO")


class SPSAEstimatorOptions(BaseEstimatorOptions):
    """
    Class for handling SPSA algorithm-specific options.
    """

    def set_algorithm_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        # Add/set algorithm configuration
        options = self.set_spsa_options(options)
        # Add/set algorithm parameter settings
        options = self.set_spsa_params(options)
        # Add/Set conditional options based on given input
        options = self.set_spsa_conditional_options(options)
        return options

    def set_spsa_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        ########################################################################
        # For the gradient approximations choose whether to use:
        #   1. Asymmetric design (one-sided difference approximation)
        #   2. Symmetric design (two-sided difference approximation)
        options["gradient_design"] = int(options.get("gradient_design", 1))
        if not options["gradient_design"] in [1, 2]:
            raise ValueError(
                "Invalid parameter value. 'gradient_design' should be 1 or 2"
            )
        ########################################################################
        # Decide the number of gradient replications to perform before
        # an estimate of the gradient is formed
        options["gradient_replications"] = int(
            options.get("gradient_replications", 1)
        )
        # Check that a value smaller than 1 has not been given
        if options["gradient_replications"] < 1:
            raise ValueError(
                "Invalid parameter value. 'gradient_replications' should be >= 1"
            )
        ########################################################################
        # Perturbation vector type:
        # Choose between:
        #   1. Random perturbations
        #   2. Deterministic perturbations
        #   3. Coordinate (equivalent to using the FDSA algorithm)
        options["perturbation_type"] = int(options.get("perturbation_type", 1))
        # Check that a value larger than 3 has not been given
        if not options["perturbation_type"] in [1, 2, 3]:
            raise ValueError(
                "Invalid parameter value. 'perturbation_type' needs to be 1, 2 or 3"
            )
        return options

    def set_spsa_params(self, options: Dict[str, Any]) -> Dict[str, Any]:
        ########################################################################
        # SPSA parameter option: Initial value for the step size
        options["param_a"] = float(options.get("param_a", 100.0))
        if options["param_a"] <= 0:
            raise ValueError("Invalid parameter value. 'param_A' should be > 0")
        ########################################################################
        # SPSA parameter option: Step size stability parameter
        options["param_A"] = float(options.get("param_A", 100.0))
        if options["param_A"] <= 0:
            raise ValueError("Invalid parameter value. 'param_A' should be > 0")
        ########################################################################
        # SPSA parameter option: Step size parameter
        options["param_alpha"] = float(options.get("param_alpha", 0.602))
        if options["param_alpha"] < 0.602 or options["param_alpha"] > 1:
            raise ValueError(
                "Invalid parameter value. 'param_alpha' should be >= 0.602 "
                + "and <= 1"
            )
        ########################################################################
        # SPSA parameter option: Initial value for the perturbation parameter
        options["param_c"] = float(options.get("param_c", 1.0))
        if options["param_c"] <= 0:
            raise ValueError("Invalid parameter value. 'param_c' should be and > 0")
        ########################################################################
        # SPSA parameter option: Perturbation parameter
        options["param_gamma"] = float(options.get("param_gamma", 0.101))
        if options["param_gamma"] < 0.101 or options["param_gamma"] > 0.166:
            raise ValueError(
                "Invalid parameter value. 'param_gamma' should be >= 0.101 "
                + "and <= 0.166"
            )
        return options

    def set_spsa_conditional_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """Set required or optional settings, that depend on user-specified input."""
        # Perturbation vector type:
        #   1. Random perturbations
        #   2. Deterministic perturbations
        #   3. Unit (equivalent to using the FDSA algorithm)
        ########################################################################
        # If deterministic perturbations (2.) were choosen, then we compute 
        # the order of the hadamard matrix used for the deterministic 
        # perturbations
        if options["perturbation_type"] == 2:
            # Gradient approximation:
            #   1. Asymmetric design (one-sided difference approximation)
            #   2. Symmetric design (two-sided difference approximation)
            if options["gradient_design"] == 1:
                matrix_order = int(2 ** np.ceil(np.log2(options["x0_size"] + 1)))
            elif options["gradient_design"] == 2:
                matrix_order = int(2 ** np.ceil(np.log2(options["x0_size"])))
            ####################################################################
            options["gradient_replications"] = 1
            hadamard_matrix = hadamard(matrix_order)
            # Store the generated hadamard matrix
            options["matrix_order"] = matrix_order
            options["hadamard_matrix"] = hadamard_matrix
        ########################################################################
        # If deterministic perturbations (3.) were choosen, then we set the 
        # number of deterministic perturbations to perform (one for each 
        # element of the gradient vector)
        if options["perturbation_type"] == 3:
            options["gradient_replications"] = options["x0_size"]
        return options


class AssignmentMatrixEstimatorOptions(BaseEstimatorOptions):

    def set_algorithm_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        return options


class SurrogateModelEstimatorOptions(BaseEstimatorOptions):
    
    def set_algorithm_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        options = self.check_required_options(options)
        return options

    def check_required_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        if "sample_directory" not in options:
            raise KeyError()
        if "n_samples" not in options:
            raise KeyError()
        else:
            print("TODO: Check non-negativity constraints")
        return options


class SumoMesoSimulationRunOptions(BaseOptions):

    def set_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        options = self.set_default_options(options)
        return options

    def set_default_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        ########################################################################
        # Set the random seed
        options["random_seed"] = int(options.get("random_seed", 1))
        ########################################################################
        options["last_step"] = int(options.get("last_step", 50))
        if options["last_step"] <= 0:
            raise ValueError(
                "Invalid parameter value. 'last_step' should be > 0"
            )
        ########################################################################
        options["convergence_steps"] = int(
            options.get("convergence-steps", options["last_step"])
        )
        if options["convergence_steps"] <= 0:
            raise ValueError(
                "Invalid parameter value. 'convergence_steps' should be > 0"
            )
        ########################################################################
        options["gA"] = float(options.get("gA", 1))
        options["gBeta"] = float(options.get("gBeta", 0.75))
        return options


class BaseDataOrganizerOptions(BaseOptions):

    def set_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        options = self.check_required_options(options)
        return options

    def check_required_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        if "seedmat" not in options:
            raise KeyError()
        return options