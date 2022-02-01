from datetime import datetime
from typing import Tuple, List, Union, Dict, Any
import json
import os

from fileserver import setup
setup()

import backend.networks.networks as networks
import backend.networks.NetworkConverter as converter
import backend.utils as ut
from backend.models import *


def read_scenario_data(config_filepath: str):
    if os.path.exists(config_filepath):    
        with open(config_filepath, mode="r") as _file:
            text_data = json.load(_file)
            file_io = FileIO._from_json(text_data["file_io"]) 
            scenario_data = ScenarioData._from_json(text_data["scenario_data"])

            # Make sure the data is saved to the database
            try:
                file_io = FileIO.objects.get(uid=file_io.uid)
            except FileIO.DoesNotExist:
                file_io.save()

            # Make sure the data is saved to the database
            try:
                scenario_data = ScenarioData.objects.get(uid=scenario_data.uid)
            except ScenarioData.DoesNotExist:
                scenario_data.save()
        return file_io, scenario_data
    else:
        raise ValueError("TODO")


def run_assignment_matrix_based_travel_demand_estimator(
    config_filepath,
    scenario_options,
    simulator_class,
    simulator_options,
    estimator_options,
    collect_directory,
    ):
    file_io, scenario_data = read_scenario_data(config_filepath=config_filepath)
    estimator = AssignmentMatrixBasedTravelDemandEstimator(
        scenario_data=scenario_data,
        file_io=file_io,
        scenario_options=scenario_options,
        simulator_class= simulator_class,
        simulator_options=simulator_options,
        estimator_options=estimator_options,
        collect_directory=collect_directory,
    )
    print("New algorithm run:")
    print(
        "scenario_options : ", scenario_options, "\n",
        "simulator_options: ", simulator_options, "\n",
        "estimator_options: ", estimator_options, "\n",
    )
    print()
    x_vec = ut.generate_constant_demand_vector(
        scenario_data=scenario_data,
        value=1.0,
    )
    x_vec = ut.construct_x_vec(
        scenario_data=scenario_data,
        demand_vec=x_vec,
    )
    estimator.run(x_vec=x_vec, collect_final_output=True)


def run_spsa_based_travel_demand_estimator(
    config_filepath,
    scenario_options,
    simulator_class,
    simulator_options,
    estimator_options,
    collect_directory,
    ) -> None:
    file_io, scenario_data = read_scenario_data(config_filepath=config_filepath)

    estimator = SPSABasedTravelDemandEstimator(
        scenario_data=scenario_data,
        file_io=file_io,
        scenario_options=scenario_options,
        simulator_class= simulator_class,
        simulator_options=simulator_options,
        estimator_options=estimator_options,
        collect_directory=collect_directory,
    )
    print("New algorithm run:")
    print(
        "scenario_options : ", scenario_options, "\n",
        "simulator_options: ", simulator_options, "\n",
        "estimator_options: ", estimator_options, "\n",
    )
    print()
    x_vec = ut.generate_constant_demand_vector(
        scenario_data=scenario_data,
        value=1.0, # 
    )
    x_vec = ut.construct_x_vec(
        scenario_data=scenario_data,
        demand_vec=x_vec,
    )
    estimator.run(x_vec=x_vec, collect_final_output=True)

def run_surrogate_model_based_travel_demand_estimator(
    config_filepath,
    scenario_options,
    simulator_class,
    simulator_options,
    estimator_options,
    collect_directory,
) -> None:
    file_io, scenario_data = read_scenario_data(config_filepath=config_filepath)
    estimator = SurrogateModelBasedTravelDemandEstimator(
        scenario_data=scenario_data,
        file_io=file_io,
        scenario_options=scenario_options,
        simulator_class=SumoMesoSimulationRun, 
        simulator_options=simulator_options,
        estimator_options=estimator_options,
        collect_directory=collect_directory,
    )
    print("New algorithm run:")
    print(
        "scenario_options : ", scenario_options, "\n",
        "simulator_options: ", simulator_options, "\n",
        "estimator_options: ", estimator_options, "\n",
    )
    print()
    x_vec = ut.generate_constant_demand_vector(
        scenario_data=scenario_data,
        value=1.0,
    )
    x_vec = ut.construct_x_vec(
        scenario_data=scenario_data,
        demand_vec=x_vec,
    )
    estimator.run(x_vec=x_vec, collect_final_output=True)

def generate_samples(
    config_filepath,
    scenario_options,
    simulator_class,
    simulator_options,
    estimator_options,
):
    file_io, scenario_data = read_scenario_data(config_filepath=config_filepath)
    sample_generator = SampleGenerator(
        scenario_data=scenario_data,
        file_io=file_io,
        simulator_class=SumoMesoSimulationRun, 
        simulator_options=simulator_options,
        scenario_options=scenario_options,
    )
    sample_generator.generate_samples(
        n_samples=int(estimator_options["max_evals"]) - 1,
        start_counter=0,
    )


def execute_all(
    config_filepath: str,
    sample_directory: str,
    simulator_options: Dict[str, Any],
    common_estimator_options: Dict[str, Any],
    weight_configurations: List[Dict[str, Any]],
    seedmats: List[str],
    collect_directory: str,
) -> None:
    ### Execute experiments related to gradient-based algorithms
    # Set possible options
    gradient_based_algorithms = ["spsa", "assignmat"]

    for algorithm in gradient_based_algorithms:
        if algorithm == "spsa":
            estimator_options = common_estimator_options.copy()
            estimator_options.update({
                "gradient_replications": 1,
                "gradient_design": 1,
                "perturbation_type": 1,
            })
            runner: Callable = \
                run_spsa_based_travel_demand_estimator
        elif algorithm == "assignmat":
            estimator_options = common_estimator_options.copy()
            runner: Callable = \
                run_assignment_matrix_based_travel_demand_estimator
        else:
            raise ValueError("TODO")
        for weight_configuration in weight_configurations:
            estimator_options["weight_configuration"] = weight_configuration
            if weight_configuration["odmat"] != 0.0:
                for seedmat in seedmats:
                    runner(
                        config_filepath=config_filepath,
                        scenario_options={"seedmat": seedmat},
                        simulator_class=SumoMesoSimulationRun,
                        simulator_options=simulator_options,
                        estimator_options=estimator_options,
                        collect_directory=collect_directory,
                    )
                    # quit()
            else:
                seedmat = seedmats[0]
                runner(
                    config_filepath=config_filepath,
                    scenario_options={"seedmat": seedmat},
                    simulator_class=SumoMesoSimulationRun,
                    simulator_options=simulator_options,
                    estimator_options=estimator_options,
                    collect_directory=collect_directory,
                )
                # quit()

    ### Execute experiments related to surrogate-based algorithms
    # Set possible options
    surrogate_models = ["fnn", "knn"]
    evaluate_output_methods = ["simdata_output"]

    runner: Callable = run_surrogate_model_based_travel_demand_estimator
    for surrogate_model in surrogate_models:
        for evaluate_output_method in evaluate_output_methods:
            for weight_configuration in weight_configurations:
                
                estimator_options = common_estimator_options.copy()
                estimator_options["weight_configuration"] = weight_configuration
                estimator_options["n_samples"] = estimator_options["max_evals"]
                estimator_options["sample_directory"] = sample_directory
                estimator_options["evaluate_output"] = evaluate_output_method
                estimator_options["method"] = surrogate_model
                if weight_configuration["odmat"] != 0.0:
                    for seedmat in seedmats:
                        runner(
                            config_filepath=config_filepath,
                            scenario_options={"seedmat": seedmat},
                            simulator_class=SumoMesoSimulationRun,
                            simulator_options=simulator_options,
                            estimator_options=estimator_options,
                            collect_directory=collect_directory,
                        )
                        # quit()
                else:
                    seedmat = seedmats[0]
                    runner(
                        config_filepath=config_filepath,
                        scenario_options={"seedmat": seedmat},
                        simulator_class=SumoMesoSimulationRun,
                        simulator_options=simulator_options,
                        estimator_options=estimator_options,
                        collect_directory=collect_directory,
                    )
                    # quit()


def generate_scenario(
    scenario_dict: Dict[str, Any],
    simulator_options: Dict[str, Any],
    func: Callable,
    func_args: Dict[str, Any],
    network_directory: str,
) -> None:
    test_scenario = SingleTestScenario(
        args_dict=scenario_dict,
        simulator_class=SumoMesoSimulationRun, 
        simulator_options=simulator_options,
        func=func,
        func_args=func_args,
        network_directory=network_directory,
    )


if "__main__" == __name__:
    print("\nExecuting...\n")
    random_seed = 111
    simulator_options = {
        "random_seed": random_seed,
        "last_step": 15,
    }
    seedmats = [
        "seedmat07",
        "seedmat09",
    ]
    weight_configurations = [
        {"odmat": 0.0, "counts": 1.0},
        {"odmat": 1.0, "counts": 1.0},
    ]
    common_estimator_options = {
        "max_evals": 200,
        "stopping_criteria": ["max_evals"],
    }

    config_filepath = "static/_09fe330b-397c-458e-9dbc-ae0c1f3c51cd/config.json"
    sample_directory = "static/_8a80589f-6047-478b-a703-30d723c66854"

    # Define which steps should be executed:
    steps = [
        # "generate_network",
        # "generate_scenario",
        # "generate_samples",
        "execute_all",
    ]
    if "generate_network" in steps:
        network_data = networks.GridNetwork(
            name="sumo-grid-network-od",
            dim=4,
            arc_length=1250,
            random_seed=random_seed,
        )
        network_data.generate_network()
        print("option::generate_network: Visualizing network...")
        fig, ax = network_data.visualize_network(
            figname="network",
            plot_type=0,
        )
        plt.savefig("network.pdf")
        cmd_args = {
            "output_dir": "input_files/network",
            "interval_increment": None,
        }
        nc = converter.NetworkConverter(
            cmd_args=cmd_args,
            network_data=network_data,
        )
        print("taz file: ", nc.taz_file)
        print("net file: ", nc.network_file)
        network_organizer = networks.SumoNetworkOrganizer(
            net_file=nc.network_file,
            taz_file=nc.taz_file,
        )
        network_organizer.initialize_network_object()
        network_organizer.generate_det_file(
            sim_output_dir=cmd_args["output_dir"],
            sim_output_path="out.xml",
            interval_increment=0.0,
        )
        network_organizer.construct_det_subset_file()
    if "generate_scenario" in steps:
        scenario_dict = {
            "interval_start": 0,
            "interval_end": 3600,
            "aggregation_intervals": 4,
            "random_seed": 321,
        }
        generate_scenario(
            scenario_dict=scenario_dict,
            simulator_options=simulator_options,
            func=ut.generate_uniform_random_demand_vector,
            func_args={"low":1.0, "high": 20.0},
            # network_directory="input",
            network_directory="input_files",
        )
    if "generate_samples" in steps:
        # Generate samples for the surrogate model based approach 
        seedmat = seedmats[0]
        estimator_options = common_estimator_options.copy()
        generate_samples(
            config_filepath=config_filepath,
            scenario_options={"seedmat": seedmat},
            simulator_class=SumoMesoSimulationRun,
            simulator_options=simulator_options,
            estimator_options=estimator_options,
        )
    if "execute_all" in steps:
        execute_all(
            config_filepath=config_filepath,
            sample_directory=sample_directory,
            simulator_options=simulator_options,
            common_estimator_options=common_estimator_options,
            weight_configurations=weight_configurations,
            seedmats=seedmats,
            collect_directory="run05",
        )
    print("\nExiting...\n")