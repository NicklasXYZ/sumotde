from typing import Union, List, Dict, Tuple, Any, Callable
from abc import abstractmethod
import logging
import uuid
import shutil
import json
import functools
import lxml
import time
import pickle
import os
from multiprocessing import Pool



from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.apps import apps
from django.conf import settings
from django.db import models
from django.utils import timezone
from SALib.sample import sobol_sequence
from SALib.util import scale_samples

from scipy.optimize import (
    OptimizeResult,
    basinhopping,
    differential_evolution,
)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor

import numpy.polynomial.polynomial as poly
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Set plotting options
sns.set_style("darkgrid")
sns.set_palette(sns.color_palette("BuGn_r"))
sns.set_context("paper")


import backend.utils as ut
from backend.EstimatorOptions import (
    SPSAEstimatorOptions,
    AssignmentMatrixEstimatorOptions,
    SurrogateModelEstimatorOptions,
    SumoMesoSimulationRunOptions,
    BaseDataOrganizerOptions,
)
from backend.fields import JSONField


# TODO: Set the 'SIMOUT_FILENAME' in the main settings file
# so it can be used across models and files
SIMOUT_FILENAME = "out"
SIMDATA_FILENAME = "simdata"
ODMAT_FILENAME = "odmat"
TRIPS_FILENAME = "trips"
GENCON_FILENAME = "gencon"
VEHROUTE_FILENAME = "vehRoutes"
TRIPINFO_FILENAME = "tripinfo"


class EstimationTaskMetaData(models.Model):
    """Database model that contains additional information about an estimation task."""

    class Meta:
        verbose_name = "Estimation Task Meta Data"
        verbose_name_plural = "Estimation Task Meta Data"
        # The "EstimationTaskMetaData" class is not an abstract base class!
        abstract = False
        ordering = ["date_time"]

    uid = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="A unique identifier.",
        verbose_name="unique identifier",
    )
    task_id = models.CharField(
        default="",
        max_length=250,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="A user-defined identifier.",
        verbose_name="task id",
    )
    date_time = models.DateTimeField(
        auto_now=True,
        auto_now_add=False,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The date and time of creation.",
        verbose_name="date time",
    )
    scenario_data = models.ForeignKey(
        "ScenarioData",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="scenario_data",
        help_text="",
        verbose_name="",
    )
    file_io = models.ForeignKey(
        "FileIO",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="file_io",
        help_text="",
        verbose_name="",
    )


class EstimationTask(models.Model):
    """Database model that contains information about an estimation task."""

    class Meta:
        verbose_name = "Estimation Task"
        verbose_name_plural = "Estimation Tasks"
        # The "EstimationTask" class is not an abstract base class!
        abstract = False

    uid = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="A unique identifier.",
        verbose_name="unique identifier",
    )
    iter_counter = models.PositiveIntegerField(
        default=0,
        help_text="An algorithm iteration counter",
        verbose_name="iteration counter",
    )
    eval_counter = models.PositiveIntegerField(
        default=0,
        help_text="An objective function evaluation counter",
        verbose_name="function evaluation counter",
    )
    metadata = models.ForeignKey(
        "EstimationTaskMetaData",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="estimation_task",
        help_text="Metadata about a particular estimation task.",
        verbose_name="metadata",
    )


class Iteration(models.Model):
    """Database model that contains information about a particular iteration of a \
    gradient-based optimization algorithm.
    """

    class Meta:
        verbose_name = "Iteration"
        verbose_name_plural = "Iterations"
        # The "Iteration" class is not an abstract base class!
        abstract = False
        ordering = ["iter_number"]

    uid = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="A unique identifier.",
        verbose_name="unique identifier",
    )
    iter_number = models.PositiveIntegerField(
        null=True,
        verbose_name="iteration number",
        help_text="The current iteration.",
    )
    x_movement = models.FloatField(
        verbose_name="x movement",
        help_text="The progress made at the current iteration: ||x - x'|| / \
            max(1, ||x||).",
        null=True,
    )
    of_saturation = models.FloatField(
        null=True,
        verbose_name="objective function saturation",
        help_text="The progress made at the current iteration: |F(x) - F(x')| / \
            max(1, |F(x)|).",
    )
    direction_magnitude = models.FloatField(
        null=True,
        verbose_name="direction mangnitude",
        help_text="The progress made at the current iteration: ||grad(F(x))||.",
    )
    estimation_task = models.ForeignKey(
        "EstimationTask",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="iterations",
        verbose_name="estimation task",
        help_text="All data pertaining to a particular estimation task.",
    )

    @property
    def iter_counter(self):
        # Back reference: Access the related 'EstimationTask' model through the
        # 'Iteration' model
        try:
            return self.estimation_task.iter_counter
        except AttributeError:
            return None

    def save(self, *args, **kwargs):
        # Set the iteration number we are currently at before saving all data
        # All data we save pertain to this iteration number
        self.iter_number = self.iter_counter
        return super().save(*args, **kwargs)


class ObjectiveFunctionEvaluation(models.Model):
    """Database model that contains information about a particular objective function \
        evaluation obtained by an optimization algorithm.
    """

    class Meta:
        verbose_name = "Objective Function Evaluation"
        verbose_name_plural = "Objective Function Evaluations"
        # The "ObjectiveFunctionEvaluation" class is not an abstract base class!
        abstract = False
        ordering = ["iter_number"]

    uid = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="A unique identifier.",
        verbose_name="unique identifier",
    )

    of = JSONField(
        null=False,
        blank=False,        
        verbose_name="objective function values",
        help_text="A dictionary of one or more objective function values.",
    )

    simulation_data = JSONField(
        null=False,
        blank=False,        
        verbose_name="",
        help_text="",
    )

    eval_number = models.PositiveIntegerField(
        verbose_name="function evaluation number.",
        help_text="The current objective function evaluation number.",
    )
    iter_number = models.PositiveIntegerField(
        null=True,
        verbose_name="iteration number",
        help_text="The current iteration number.",
    )
    wall_time = models.FloatField(
        verbose_name="wall clock time",
        help_text="The wall clock time it took to evaluate the objective function.",
    )
    sysusr_time = models.FloatField(
        verbose_name="system+user time",
        help_text="The system+user time it took to evaluate the objective function.",
    )
    take_step = models.BooleanField(
        default=False,
        null=False,
        verbose_name="take step",
        help_text="A value indicating if the objective function evaluation was made \
            at the very end of an iteration. That is, the value indicates if the \
            objective function value is obtained from the adjusted solution: x_new \
            = x_old -gradient * step-length.",
    )
    estimation_task = models.ForeignKey(
        "EstimationTask",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="of_evals",
        verbose_name="objective function evaluations",
        help_text="All data pertaining to a particular estimation task.",
    )
    iteration = models.ForeignKey(
        "Iteration",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="iteration_of_evaluations",
        verbose_name="iteration",
        help_text="All data pertaining to a particular iteration of the alg.",
    )

    @property
    def eval_counter(self):
        # Back reference: Access the related 'EstimationTask' model through the
        # 'DescentDirection' model
        try:
            return self.estimation_task.eval_counter
        except AttributeError:
            pass

    @property
    def iter_counter(self):
        # Back reference: Access the related 'EstimationTask' model through the
        # 'ObjectiveFunctionEvaluation' model
        try:
            return self.estimation_task.iter_counter
        except AttributeError:
            return None

    def save(self, *args, **kwargs):
        # Set the objective function evaluation counter and the iteration number we
        # are currently at before saving all data
        # All data we save pertain to this objective function evaluation number and
        # iteration
        self.eval_number = self.eval_counter
        self.iter_number = self.iter_counter
        return super().save(*args, **kwargs)


class FileIO(models.Model):

    class Meta:
        verbose_name = "FileIO"
        verbose_name_plural = "FileIO"
        # The "FileIO" class is not an abstract base class!
        abstract = False
        ordering = ["date_time"]

    uid = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="A unique identifier.",
        verbose_name="unique identifier",
    )
    output_directory_identifier = models.CharField(
        default="",
        max_length=250,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="",
        verbose_name="",
    )
    date_time = models.DateTimeField(
        auto_now=True,
        auto_now_add=False,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="The date and time of creation.",
        verbose_name="date time",
    )
    root_input_directory = models.CharField(
        default="",
        max_length=250,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        help_text="",
        verbose_name="",
    )
    directory_counter = models.PositiveIntegerField(
        default=0,
        help_text="",
        verbose_name="",

    )

    def output_directory(self, new: bool = False) -> str:
        output_directory = os.path.join(
            self.root_output_directory,
            str(self.directory_counter),
        )
        # If no new directory should be created (new == False), then return 
        # the one currently used
        if new is False:
            return output_directory
        # If a new directory should be created (new == True), then first check 
        # if a directory already exists with the current directory count and 
        # create it. Otherwise increment the directory counter and create a 
        # new directory
        else:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok=True)
                return output_directory
            else:
                self.directory_counter += 1
                output_directory = os.path.join(
                    self.root_output_directory,
                    str(self.directory_counter),
                )
                os.makedirs(output_directory, exist_ok=True)
                # Update database 
                self.save()
                return output_directory

    @functools.cached_property
    def root_output_directory(self) -> str:
        _path = os.path.join(
            settings.MEDIA,
            f"{self.output_directory_identifier}_{self.uid}",
        )
        return _path

    def save(self, *args, **kwargs):
        if not os.path.exists(self.root_output_directory):
            os.makedirs(self.root_output_directory, exist_ok=True)
        return super().save(*args, **kwargs)

    def _to_json(self) -> str:
        contents = {
            "uid": str(self.uid),
            "output_directory_identifier": self.output_directory_identifier,
            "date_time": str(self.date_time),
            "root_input_directory": self.root_input_directory,
            "directory_counter": str(self.directory_counter),
        }
        return json.dumps(contents, indent=4, sort_keys=True)

    @classmethod
    def _from_json(cls, json_data: str) -> "FileIO":
        # TODO: If uid is passed make sure to retrieve the data from db
        # instead of re-creating the data
        data_in = json.loads(json_data)
        data_in["directory_counter"] = ut.str_to_int(
            data_in["directory_counter"]
        )
        fio = FileIO(**data_in)
        return fio



class SumoNetwork(models.Model):

    network_files = JSONField(
        default=None,
        null=True,
        blank=True,
        help_text="",
        verbose_name="",
    )

    # TODO: Create a django model serializer instead
    def _to_json(self):
        # TODO: Make sure the 'network_files' paths are absolute paths
        # do they can be located from anywhere
        if isinstance(self.network_files, dict):
            # TODO: Just root input directory is enough
            return json.dumps(self.network_files, indent=4, sort_keys=True)
        else:
            return self.network_files

    # TODO: Create a django model serializer instead
    @classmethod
    def _from_json(cls, json_data: str):
        # TODO: If uid is passed make sure to retrieve the data from db
        # instead of re-creating the data
        # TODO: Just root input directory is enough
        data_in = json.loads(json_data)

        _dir = os.path.dirname(data_in["net"])
        tmp_dir = _dir
        while True:
            tmp_dir = os.path.dirname(tmp_dir)        
            if tmp_dir != "":
                _dir = os.path.dirname(_dir)
            else:
                break
        sn = SumoNetwork(**{})
        sn._read_network_files(root_input_directory=_dir)
        return sn

    def _read_network_files(
        self,
        root_input_directory: str,
        ) -> None:
        if os.path.exists(root_input_directory) is True:
            network_file_directory = os.path.join(
                root_input_directory,
                "network"
            )
            network_files = {}
            for _file in os.listdir(network_file_directory):
                split_string = _file.split(".")
                if "net" in split_string:
                    network_files["net"] = os.path.join(
                        network_file_directory,
                        _file,
                    )
                elif "taz" in split_string:
                    network_files["taz"] = os.path.join(
                        network_file_directory,
                        _file,
                    )
                elif "det" in split_string:
                    network_files["det"] = os.path.join(
                        network_file_directory,
                        _file,
                    )
                elif "det_subset" in split_string:
                    network_files["det_subset"] = os.path.join(
                        network_file_directory,
                        _file,
                    )
            required_keys = ["net", "taz", "det", "det_subset"]
            for key in network_files.keys():
                if key not in required_keys:
                    raise ValueError("TODO")
            self.network_files = json.dumps(
                network_files,
                indent=4,
                sort_keys=True
            )

    def get_nodes(self, root: lxml.etree._Element) -> pd.DataFrame:
        nodes: List[Dict[str, Any]] = []
        for child in root:
            if child.tag == "junction":
                junction_attributes = {}
                if child.attrib["type"] != "internal":
                    for attribute in child.attrib:
                        junction_attributes[attribute] = child.attrib[attribute]
                    nodes.append(junction_attributes)
        return pd.DataFrame(data=nodes, dtype=object)
    
    def get_links(self, root: lxml.etree._Element) -> pd.DataFrame:
        links: List[Dict[str, Any]] = []
        for child in root:
            if ((child.tag == "edge" and "priority" in child.attrib) and \
            ("from" in child.attrib and "to" in child.attrib)):
                link_attributes: Dict[str, Any] = {}
                link_attributes["link"] = child.attrib["id"]
                # link_attributes["edge"] = child.attrib["id"]
                link_attributes["from"] = child.attrib["from"]
                link_attributes["to"] = child.attrib["to"]
                link_attributes["priority"] = child.attrib["priority"]
                lane_id = []; lane_index=[]; lane_length = []; lane_speed = []
                for item in child:
                    if item.tag == "lane":
                        lane_id.append(item.attrib["id"])
                        lane_index.append(item.attrib["index"])
                        lane_length.append(item.attrib["length"])
                        lane_speed.append(item.attrib["speed"])
                    if item.tag == "param":
                        link_attributes[item.attrib["key"]] = item.attrib["value"]
                link_attributes["lane"] = lane_id
                link_attributes["index"] = lane_index
                link_attributes["length"] = lane_length
                link_attributes["speed"] = lane_speed
                links.append(link_attributes)
        _types = {"from": str, "to": str, "link": str}
        return pd.DataFrame(data=links, dtype=object).astype(_types)

    def get_det(self, root: lxml.etree._Element) -> pd.DataFrame:
        det: List[Dict[str, Any]] = []
        for child in root:
            if child.tag == "inductionLoop":
                det_attribs = {}
                for attribute in child.attrib:
                    det_attribs[attribute] = child.attrib[attribute]
                det.append(det_attribs)
        return pd.DataFrame(data=det, dtype=object)

    def get_det_subset(self, root: lxml.etree._Element) -> pd.DataFrame:
        det_subset: List[Dict[str, Any]] = []
        for child in root:
            if child.tag == "additional":
                for subchild in child:
                    if subchild.tag == "inductionLoop":
                        det_subset_attribs = {}
                        for attribute in subchild.attrib:
                            if attribute == "id":
                                det_subset_attribs["link"] = \
                                    subchild.attrib["id"]
                            else:
                                det_subset_attribs[attribute] = \
                                    subchild.attrib[attribute]
                        det_subset.append(det_subset_attribs)
        return pd.DataFrame(data=det_subset, dtype=object)

    def get_taz(self, root: lxml.etree._Element) -> pd.DataFrame:
        taz: List[Dict[str, Any]] = []
        for child in root:
            if child.tag == "taz":
                taz_attribs = {}; sources = ""; sinks = ""
                taz_attribs["id"] = child.attrib["id"]
                taz_attribs["sources"] = ""
                taz_attribs["sinks"] = ""
                for nestedChild in child:
                    if nestedChild.tag == "tazSource":
                        sources += nestedChild.attrib["id"] + " "
                    elif nestedChild.tag == "tazSink":
                        sinks += nestedChild.attrib["id"] + " "
                if len(sources) != 0 or len(sinks) != 0:
                    taz_attribs["sources"] = sources.strip()
                    taz_attribs["sinks"] = sinks.strip()
                elif "edges" in child.attrib:
                    taz_attribs["sources"] += child.attrib["edges"].strip()
                    taz_attribs["sinks"] += child.attrib["edges"].strip()
                if taz_attribs["sources"] != "":
                    taz_attribs["sources"] = taz_attribs["sources"].split(" ")
                if taz_attribs["sinks"] != "":
                    taz_attribs["sinks"] = taz_attribs["sinks"].split(" ")
                taz.append(taz_attribs)
        return pd.DataFrame(data=taz, dtype=object)

    @functools.cached_property
    def nodes(self) -> pd.DataFrame:
        if self.network_files is not None:
            if isinstance(self.network_files, str) or \
                isinstance(self.network_files, bytearray):
                network_files = json.loads(self.network_files)
            else:
                network_files = self.network_files
            if isinstance(network_files, dict):
                root = ut.read_xml_file(
                    xml_file=network_files["net"],
                )
                nodes = self.get_nodes(root=root)
                return nodes
            else:
                raise ValueError(
                    "Required object attribute 'network_files' is not of type \
                    'dict'. Attribute 'nodes' could thus not be loaded."
                )
        else:
            raise ValueError(
                "Object attribute 'network_files' is 'None'."
            )

    @functools.cached_property
    def links(self) -> pd.DataFrame:
        if self.network_files is not None:
            if isinstance(self.network_files, str) or \
                isinstance(self.network_files, bytearray):
                network_files = json.loads(self.network_files)
            else:
                network_files = self.network_files
            if isinstance(network_files, dict):
                root = ut.read_xml_file(
                    xml_file=network_files["net"],
                )
                links = self.get_links(root=root)
                return links
            else:
                raise ValueError(
                    "Required object attribute 'network_files' is not of type \
                    'dict'. Attribute 'links' could thus not be loaded."
                )
        else:
            raise ValueError(
                "Object attribute 'network_files' is 'None'."
            )
    
    @functools.cached_property
    def det(self) -> pd.DataFrame:
        if self.network_files is not None:
            if isinstance(self.network_files, str) or \
                isinstance(self.network_files, bytearray):
                network_files = json.loads(self.network_files)
            else:
                network_files = self.network_files
            if isinstance(network_files, dict):
                root = ut.read_xml_file(
                    xml_file=network_files["det"],
                )
                links = self.get_det(root=root)
                return links
            else:
                raise ValueError(
                    "Required object attribute 'network_files' is not of type \
                    'dict'. Attribute 'det' could thus not be loaded."
                )
        else:
            raise ValueError(
                "Object attribute 'network_files' is 'None'."
            )

    @functools.cached_property
    def det_subset(self) -> pd.DataFrame:
        if self.network_files is not None:
            if isinstance(self.network_files, str) or \
                isinstance(self.network_files, bytearray):
                network_files = json.loads(self.network_files)
            else:
                network_files = self.network_files
            if isinstance(network_files, dict):
                root = ut.read_xml_file(
                    xml_file=network_files["det_subset"],
                )
                links = self.get_det_subset(root=root)
                return links
            else:
                raise ValueError(
                    "Required object attribute 'network_files' is not of type \
                    'dict'. Attribute 'det_subset' could thus not be loaded."
                )
        else:
            raise ValueError(
                "Object attribute 'network_files' is 'None'."
            )
    
    @functools.cached_property
    def taz(self) -> pd.DataFrame:
        if self.network_files is not None:
            if isinstance(self.network_files, str) or \
                isinstance(self.network_files, bytearray):
                network_files = json.loads(self.network_files)
            else:
                network_files = self.network_files
            if isinstance(network_files, dict):
                root = ut.read_xml_file(
                    xml_file=network_files["taz"],
                )
                taz = self.get_taz(root=root)
                return taz
            else:
                raise ValueError(
                    "Required object attribute 'network_files' is not of type \
                    'dict'. Attribute 'taz' could thus not be loaded."
                )
        else:
            raise ValueError(
                "Object attribute 'network_files' is 'None'."
            )

    @functools.cached_property
    def _get_origins_and_destinations(self) -> Tuple[
        List[Any], List[Any], int, Dict[str, Any]
        ]:
        _sources = []; _sinks = []; _sources_and_sinks_counter = 0
        taz_to_dict = {}
        for taz, sources, sinks in self.taz.values:
            if isinstance(sources, list) and isinstance(sinks, list):
                if len(sources) == 1 and len(sinks) == 1:
                    _sources_and_sinks_counter += 1
                    _sources += sources
                    _sinks += sinks
                elif len(sources) == 0 and len(sinks) == 1:
                    _sinks += sinks
                elif len(sources) == 1 and len(sinks) == 0:
                    _sources += sources       
                else:
                    raise ValueError(
                        f"Currently, multiple source or sink links can not be \
                        associated with the same TAZ.\n The TAZ {taz} had the \
                        following links associated with it: \n \
                        Source links: {sources} \n \
                        Sink links  : {sinks}"
                    )
                try:
                    taz_to_dict[taz]["sources"].extend(sources)
                    taz_to_dict[taz]["sinks"].extend(sinks)
                except KeyError:
                    taz_to_dict[taz] = {
                        "sources": sources,
                        "sinks": sinks,
                    }
            else:
                raise ValueError("TODO")
        if set(_sources).issubset(_sinks):
            raise ValueError(
                "A link can not simultaneously be a source and a sink."
            )
        n_sources = len(_sources); n_sinks = len(_sinks)
        if n_sources != len(set(_sources)):
            raise ValueError(
                "Make sure source links do not appear more than once in any \
                TAZ or across different TAZ."
            )
        if n_sinks != len(set(_sinks)):
            raise ValueError(
                "Make sure sink links do not appear more than once in any \
                TAZ or across different TAZ."
            )
        return _sources, _sinks, _sources_and_sinks_counter, taz_to_dict

    @functools.cached_property
    def taz_to_dict(self) -> Dict[str, Any]:
        _, _, _, taz_to_dict = self._get_origins_and_destinations 
        return taz_to_dict

    @functools.cached_property
    def origins(self) -> List[Any]:
        _origins, _, _, _ = self._get_origins_and_destinations 
        return _origins
    
    @functools.cached_property
    def destinations(self) -> List[Any]:
        _, _destinations, _, _ = self._get_origins_and_destinations 
        return _destinations

    @functools.cached_property
    def total_od_pairs(self) -> int:
        _sources, _sinks, _sources_and_sinks, _ = \
            self._get_origins_and_destinations
        return len(_sources) * len(_sinks) - _sources_and_sinks
    
    def save(self, *args, **kwargs):
        if self.network_files is None or \
            self.network_files == "":
            raise ValueError(
                "The 'network_files' attribute has not yet been set. Call the \
                '_read_network_files' method with an appropriate directory."
            )
        return super().save(*args, **kwargs)
    

class ScenarioData(models.Model):

    class Meta:
        verbose_name = "Scenario Data"
        verbose_name_plural = "Scenario Data"
        # The "ScenarioData" class is not an abstract base class!
        abstract = False

    uid = models.UUIDField(
        primary_key=True,
        default=uuid.uuid4,
        editable=False,
        null=False,
        blank=False,
        unique=True,
        help_text="A unique identifier.",
        verbose_name="unique identifier",
    )
    random_seed = models.IntegerField(
        editable=False,
        null=False,
        blank=False,
        unique=False,
        help_text="",
        verbose_name="",
    )
    interval_start = models.PositiveIntegerField(
        editable=False,
        null=False,
        blank=False,
        unique=False,
        help_text="",
        verbose_name="",
    )
    interval_end = models.PositiveIntegerField(
        editable=False,
        null=False,
        blank=False,
        unique=False,
        help_text="",
        verbose_name="",
    )
    aggregation_intervals = models.PositiveIntegerField(
        editable=False,
        null=False,
        blank=False,
        unique=False,
        help_text="",
        verbose_name="",
    )
    interval_increment = models.IntegerField(
        default=None,
        editable=False,
        null=True,
        blank=False,
        unique=False,
        help_text="",
        verbose_name="",
    )
    network = models.ForeignKey(
        "SumoNetwork",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="scenario_data",
        verbose_name="",
        help_text="",
    )

    @functools.cached_property
    def intervals(self):
        _intervals = [
            [
                self.interval_start + (i - 1) * self.interval_increment,
                self.interval_start + i * self.interval_increment
            ] for i in range(1, self.aggregation_intervals + 1)
        ]
        return _intervals

    def save(self, *args, **kwargs):
        if self.interval_increment is None:
            self.interval_increment = int(
                (self.interval_end - self.interval_start) / \
                self.aggregation_intervals
            )
        if self.network is not None:
            self.network.save()
        return super().save(*args, **kwargs)

    # TODO: Create django model serializers
    def _to_json(self) -> str:
        contents = {
            "random_seed": str(self.random_seed),
            "interval_start": str(self.interval_start),
            "interval_end": str(self.interval_end),
            "aggregation_intervals": str(self.aggregation_intervals),
            "interval_increment": str(self.interval_increment),
            "network": self.network._to_json(),
        }
        return json.dumps(contents, indent=4, sort_keys=True)
    
    @classmethod
    def _from_json(cls, json_data: str) -> "ScenarioData":
        data_in = json.loads(json_data)
        data_in["random_seed"] = ut.str_to_int(data_in["random_seed"])
        data_in["interval_start"] = float(data_in["interval_start"])
        data_in["interval_end"] = float(data_in["interval_end"])
        data_in["aggregation_intervals"] = \
            ut.str_to_int(data_in["aggregation_intervals"])
        data_in["interval_increment"] = \
            float(data_in["interval_increment"])

        data_in["network"] = SumoNetwork()._from_json(data_in["network"])
        scenario_data = ScenarioData(**data_in)
        return scenario_data


class SumoMesoSimulationRun(models.Model):

    simulation_files = JSONField(
        null=True,
        blank=True,
        help_text="",
        verbose_name="",
    )
    file_io = models.ForeignKey(
        "FileIO",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="sumo_meso_simulation_runs",
        verbose_name="",
        help_text="",
    )
    scenario_data = models.ForeignKey(
        "ScenarioData",
        on_delete=models.CASCADE,
        editable=True,
        null=True,
        blank=True,
        unique=False,
        related_name="sumo_meso_simulation_runs",
        verbose_name="",
        help_text="",
    )
    simulator_options = JSONField(
        null=False,
        blank=False,        
        verbose_name="",
        help_text="",
    )
    wall_time = models.FloatField(
        verbose_name="wall clock time",
        help_text="The wall clock time it took to run the simulator",
    )
    sysusr_time = models.FloatField(
        verbose_name="system+user time",
        help_text="The system+user time it took to evaluate the simulator",
    )

    @staticmethod
    def get_options_handler():
        return SumoMesoSimulationRunOptions

    def generate_travel_demand_file(
        self,
        x_vec: Union[None, pd.DataFrame],
        ) -> pd.DataFrame:
        travel_demand_file = os.path.join(
            self.file_io.output_directory(new=True),
            f"{TRIPS_FILENAME}.xml"
        )
        odmat_file = os.path.join(
            self.file_io.output_directory(),
            f"{ODMAT_FILENAME}.csv"
        )
        self._generate_travel_demand_file(
            x_vec=x_vec,
            travel_demand_file=travel_demand_file,
            odmat_file=odmat_file,
        )

        # Save paths to simulation files
        dict_ = {
            "odmat": odmat_file,
        }
        print("odmat self.simulation_files: ", self.simulation_files)
        if self.simulation_files is None:
            self.simulation_files = dict_
        elif isinstance(self.simulation_files, Dict):
            if len(self.simulation_files) == 0:
                self.simulation_files.update(dict_)
            else:
                raise ValueError("TODO")
        else:
            raise ValueError("TODO")
        print("odmat self.simulation_files: ", self.simulation_files)


    def _generate_travel_demand_file(
        self,
        x_vec: pd.DataFrame,
        travel_demand_file: str,
        odmat_file: str,
        ) -> None:
        if isinstance(x_vec, pd.DataFrame):
            # Open a new output file.
            f = open(travel_demand_file, "w")
            # Write out the prolog together with the link to the XML schema
            f.write("<?xml version=\"1.0\" ?>\n")
            f.write(
                "<routes " + \
                "xmlns:xsi=" + \
                "\"http://www.w3.org/2001/XMLSchema-instance\" " + \
                "xsi:noNamespaceSchemaLocation=" + \
                "\"http://sumo.dlr.de/xsd/routes_file.xsd\"" + \
                ">\n"
            )
            # Write to the filestream the contents of a 'trip.xml' file that
            # can be read by SUMO
            self._trips_content(f=f, x_vec=x_vec)
            f.write("</routes>\n")
            f.close()

            # ... Also save the trips as a pandas dataframe in a .csv file
            x_vec.to_csv(odmat_file)
        else:
            raise ValueError("TODO")

    def _trips_content(self, f, x_vec: pd.DataFrame) -> None:
        # Locate all origin-destinaiton pairs that have been assigned a trip 
        # value larger than 0
        _x_vec = x_vec.loc[x_vec["value"] > 0]
        # Define empty string we can append to
        str_out = ""
        for _, row in _x_vec.iterrows():
            str_out += self._write_trip(row=row)
        # Write the generated string to the open file 
        f.write(str_out)

    def _write_trip(self, row: pd.Series) -> str:
        # Take the first origin link in the list contained in the \
        # dataframe containing TAZ source and sink links 
        origin = self.scenario_data.network.taz_to_dict[
            row["from"]
        ]["sources"][0]
        # Take the first destination link in the list contained in the \
        # dataframe containing TAZ source and sink links 
        destination = self.scenario_data.network.taz_to_dict[
            row["to"]
        ]["sinks"][0]

        # TODO: Do 'value' need to me rounded?
        # TODO: If necessary, then apply column-wise, instead of row-wise here
        value = int(np.round(row["value"]))
        return self._trip_tag(
            id_=str(row.name),
            number_=str(value),
            begin_=str(row["begin"]),
            end_=str(row["end"]),
            from_=str(origin),
            to_=str(destination),
            fromTaz_=str(row["from"]),
            toTaz_=str(row["to"]),
        )

    def _trip_tag(
        self,
        id_: str,
        number_: str,
        begin_: str,
        end_: str,
        from_: str,
        to_: str,
        fromTaz_: str,
        toTaz_: str,
        departLane_: str = "free",
        departSpeed_: str = "max",
        ) -> str:
        """
        The number of inserted vehicles (if "no" is not given) is equal to
        ("end"-"begin")/"period" rounded to the nearest integer, thus if 
        "period" is small enough, there might be no vehicle at all. 
        Furthermore "period"=3600/"vehsPerHour".
        """
        return f'\t<flow' + \
            f' id=\"{id_}\" ' + \
            f' number=\"{number_}\"' \
            f' begin=\"{begin_}\"' \
            f' end=\"{end_}\"' + \
            f' from=\"{from_}\"' + \
            f' to=\"{to_}\"' + \
            f' fromTaz=\"{fromTaz_}\"' + \
            f' toTaz=\"{toTaz_}\"' + \
            f' departLane=\"{departLane_}\"' + \
            f' departSpeed=\"{departSpeed_}\"/>\n'

    def _sumo_cli_args(self, parse_tripinfos: bool = False) -> str:
        if self.scenario_data.network.network_files is not None:
            if isinstance(self.scenario_data.network.network_files, str) or \
                isinstance(self.scenario_data.network.network_files, bytearray):
                network_files = json.loads(
                    self.scenario_data.network.network_files,
                )
            else:
                network_files = self.scenario_data.network.network_files
            if isinstance(network_files, dict):
                # Construct the relative filepath from the given input network
                # file to the currently used output directory 
                relative_path_to_output_dir = os.path.relpath(
                    os.path.dirname(network_files["net"]),
                    self.file_io.output_directory(),
                )
                # NOTE: The path to the '*.net.xml' file should be
                #       specified as a relative path to a '.sumocfg' file 
                #       generated by the 'duaIterate.py' python script
                net_file = os.path.join(
                    relative_path_to_output_dir,
                    os.path.basename(network_files["net"])
                )
                
                # NOTE: The 'vehRoute_file.veh.xml' file is saved relative to 
                #       a '.sumocfg' file generated by the 'duaIterate.py'
                #       python script
                # NOTE: Make sure to only derive necessary data from the output 
                #       '*.veh.xml' file.
                #       It seems that the data recorded in this file might not
                #       be consistent with other types output generated by the
                #       simulator. For example, it might be recorded in the 
                #       '*.veh.xml' file that a vehicle exits a link at a 
                #       certain time. However, a simulated detector placed at 
                #       the very end of a link might not record and count this
                #       vehicle 
                vehRoute_file = f"{VEHROUTE_FILENAME}.veh.xml"

                # Generate string of commandline arguments that should be passed
                # to the 'duaIterate.py' python script 
                # NOTE: Do not use routing-threads else duarouter gets stuck...
                executable = "python /usr/share/sumo/tools/assign/duaIteratev3.py"
                begin = int(self.scenario_data.interval_start)
                end = int(self.scenario_data.interval_end) + \
                    int(self.scenario_data.interval_increment) 
                time_to_teleport = int(self.scenario_data.interval_end) + \
                    int(self.scenario_data.interval_increment)

                # Use the given random seed for reproducibility purposes
                commandline_args = \
                    f"{executable} " + \
                    f"--net-file {net_file} " + \
                    f"--trips trips.xml " + \
                    f"--last-step {str(self.simulator_options['last_step'])} " + \
                    f"--convergence-steps {str(self.simulator_options['convergence_steps'])} " + \
                    f"--output-path {self.file_io.output_directory()} " + \
                    f"--begin {str(begin)} " + \
                    f"--end {str(end)} " + \
                    f"--mesosim " + \
                    f"--time-to-teleport {str(time_to_teleport)} "
                    # f"--gA {str(self.simulator_options['gA'])} " + \
                    # f"--gBeta {str(self.simulator_options['gBeta'])} " + \
                # TODO: Create flag: Enable when creating the test scenario 
                #       files
                if parse_tripinfos is False:
                    commandline_args += "--disable-tripinfos --disable-summary "
                # "duarouter--bulk-routing " + \
                # commandline_args += \
                #     f"duarouter--seed {str(self.simulator_options['random_seed'])} " + \
                #     "duarouter--remove-loops " + \
                #     "duarouter--xml-validation " + "never" + " " + \
                #     "sumo--xml-validation " + "never" + " " + \
                #     "sumo--vehroute-output " + vehRoute_file + " " + \
                #     "sumo--vehroute-output.sorted " + \
                #     "sumo--vehroute-output.exit-times " + \
                #     "sumo--vehroute-output.last-route " + \
                #     "sumo--vehroute-output.write-unfinished " + \
                #     "sumo--default.speeddev 0 " + \
                #     "sumo--meso-junction-control " + \
                #     "sumo--max-depart-delay " + \
                #         str(
                #             int(self.scenario_data.interval_end) + \
                #             int(self.scenario_data.interval_increment)
                #         )
                commandline_args += \
                    f"duarouter--seed {str(self.simulator_options['random_seed'])} " + \
                    "duarouter--remove-loops " + \
                    "duarouter--xml-validation " + "never" + " " + \
                    "sumo--xml-validation " + "never" + " " + \
                    "sumo--vehroute-output " + vehRoute_file + " " + \
                    "sumo--vehroute-output.sorted " + \
                    "sumo--vehroute-output.exit-times " + \
                    "sumo--vehroute-output.last-route " + \
                    "sumo--vehroute-output.write-unfinished " + \
                    "sumo--default.speeddev 0 " + \
                    "sumo--meso-junction-control " + \
                    "sumo--max-depart-delay " + \
                        str(
                            int(self.scenario_data.interval_end) + \
                            int(self.scenario_data.interval_increment)
                        )
                print("Commandline args: ", commandline_args)
                return commandline_args
            else:
                raise ValueError("TODO")
        else:
            raise ValueError("TODO")

    def _cleanup(self) -> None:
        # Delete the generated files
        output_dir = self.file_io.output_directory()
        _dirs = [
            directory for directory in os.listdir(output_dir) \
            if os.path.isdir(
                os.path.join(output_dir, directory)
            ) is True
        ]
        _dirs = sorted(_dirs, key=lambda x: int(x))
        _dirs = _dirs[:-1]
        for directory in _dirs:
            dir_path = os.path.join(output_dir, directory)
            shutil.rmtree(dir_path, ignore_errors=True)

    def _handle_no_flow(self, x_vec: Union[None, pd.DataFrame]):
        df1 = pd.DataFrame(
            columns=[
                "depart",
                "from", "to",
                "counts",
                "link",
                "begin", "end",
                "exitTime",
            ]
        )
        # Determine the usage of: 
        # - Each of the links in each of the time intervals
        cols = ["begin", "end", "link", "counts"]
        df2 = df1[cols]

        # - Each of the links in each of the time intervals by each origin-
        #   destination pair in each of the time intervals
        cols = ["begin", "end", "link", "from", "to", "depart", "counts"]
        df3 = df1[cols]

        df1, df2, df3 = self._organize_simdata(
            x_vec=x_vec,
            df1=df1,
            df2=df2,
            df3=df3,
        )
        return df1, df2, df3, None

    @ut.timing
    def run_simulator(
        self,
        x_vec: pd.DataFrame,
        cleanup: bool = True,
        parse_tripinfos: bool = False,
        ):
        total_demand = x_vec["value"].round().sum()
        if total_demand != 0:
            df1, df2, df3, dat = self._run_simulator(
                x_vec=x_vec,
                parse_tripinfos=parse_tripinfos,
            )
            if cleanup is True:
                self._cleanup()
        else:
            print(f"All trip demands are 0") 
            # logging.info(f"All trip demands are 0") 
            df1, df2, df3, dat = self._handle_no_flow(x_vec=x_vec)
        return df1, df2, df3, dat

    def _organize_simdata(
        self,
        x_vec: pd.DataFrame,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df3: pd.DataFrame,
        ) -> Tuple[pd.DataFrame,...]:
        # Construct an approximation of the assignment matrix.
        # Merge on the origin and destination along with the start of the 
        # interval in which demand has been assigned. In 'df3' the start of the
        # interval is stored in column 'depart' so rename the column to align
        # the column names of 'x_vec' and 'df3' 
        df3 = df3.rename(
            columns={
                "begin": "link_begin",
                "depart": "begin",
                "end": "link_end",
            },
        )
        df3 = df3.merge(
            x_vec[["begin", "from", "to", "value"]],
            on=["from", "to", "begin"],
            how="left",
        )
        # TODO: Round trip demands to make sure ratios are within [0, 1]
        #       Counts will be integer valued as SUMO (internally) reads
        #       the trip demands as integer values as well
        df3["ratio"] = df3["counts"] / df3["value"].round()
        # df3["ratio"] = df3["counts"] / df3["value"]
        return df1, df2, df3
    
    def _get_tripinfos_file(self):
        last_step = int(self.simulator_options["last_step"]) - 1
        tripinfos_file = os.path.join(
            self.file_io.output_directory(),
            str(last_step), 
            f"{TRIPINFO_FILENAME}_{last_step:03d}.xml",
        )
        if os.path.exists(tripinfos_file):
            return tripinfos_file
        # Try to search for the correct directory
        else:
            # dirs = os.listdir(self.file_io.output_directory())
            # _dirs = []
            # for dir in dirs:
            #     if ut.str_is_number(dir):
            #         _dirs.append(ut.str_to_int(dir))
            # max_value = np.max(_dirs)
            # vehRoute_file = os.path.join(
            #     # self.file_io.root_output_directory,
            #     self.file_io.output_directory(),
            #     f"{max_value}", # TODO: Based on number of dua iterations
            #     f"{VEHROUTE_FILENAME}.veh.xml",
            # )
            # return vehRoute_file
            raise ValueError("TODO")

    def _parse_tripinfos(self) -> pd.DataFrame:
        end = int(self.scenario_data.interval_end) + \
            int(self.scenario_data.interval_increment) 
        tripinfos_file = self._get_tripinfos_file()
        root = ut.read_xml_file(xml_file=tripinfos_file)
        _dict_data = []
        # for child in root:
        #     if child.tag == "tripinfo":
        #         tripinfo = {}
        #         if (int(float(child.attrib["depart"])) <= \
        #             self.scenario_data.intervals[-1][1]):
        #             for attribute in child.attrib:
        #                 tripinfo[attribute] = child.attrib[attribute]
        #             _dict_data.append(tripinfo)
        for child in root:
            if child.tag == "tripinfo":
                tripinfo = {}
                if int(float(child.attrib["depart"])) <= end:
                    for attribute in child.attrib:
                        tripinfo[attribute] = child.attrib[attribute]
                    _dict_data.append(tripinfo)
        _types = {
            "id": str,
            "depart": float,
            "departLane": str,
            "departPos": float,
            "departSpeed": float,
            "departDelay": float,
            "arrival": float,
            "arrivalLane": str,
            "arrivalPos": float,
            "arrivalSpeed": float,
            "duration": float,
            "routeLength": float,
            "waitingTime": float,
            "waitingCount": float,
            "stopTime": float,
            "timeLoss": float,
            "rerouteNo": float,
            "devices": str,
            "vType": str,
            "speedFactor": float,
            "vaporized": str,
        }
        df = pd.DataFrame(data=_dict_data, dtype=object).astype(dtype=_types)
        _file = os.path.join(
            self.file_io.output_directory(),
            f"{TRIPINFO_FILENAME}.csv",
        )
        df.to_csv(_file)
        return df

    def _run_simulator(
        self,
        x_vec: pd.DataFrame,
        parse_tripinfos: bool = False
        ) -> Tuple[pd.DataFrame,...]:
        print("Generating sumo input files before running simulations...")
        self.generate_travel_demand_file(x_vec=x_vec)
        commandline_args = self._sumo_cli_args(
            parse_tripinfos=parse_tripinfos,
        )
        # Timing start
        start_time_sim_wall = time.time()
        start_time_sim_sysusr = time.process_time()
        # CALL SIMULATOR
        ut.run_subprocess(commandline_args)
        # Timing end
        end_time_sim_wall = time.time()
        end_time_sim_sysusr = time.process_time()
        # Calculate duration of the simulations
        self.wall_time = end_time_sim_wall - start_time_sim_wall
        self.sysusr_time = end_time_sim_sysusr - start_time_sim_sysusr

        # Read and organize simulation data
        df1, df2, df3 = self._read_simdata(x_vec=x_vec)
        df1, df2, df3 = self._organize_simdata(
            x_vec=x_vec,
            df1=df1,
            df2=df2,
            df3=df3,
        )
        if parse_tripinfos is True:
            self._parse_tripinfos()
           
        # dat = self._parse_dua_log()
        # self._parse_summary()

        # df1    : Raw vehRoute data 
        # df1    : Link counts
        # df3    : Assignment matrix data
        # None   : None
        return df1, df2, df3, None

    def _read_simdata(self, x_vec: pd.DataFrame) -> pd.DataFrame:
        # TODO: Remove counts that are not equipped with a detector
        df1 = self._read_vehRoute_file(x_vec=x_vec)
        _file = os.path.join(
            self.file_io.output_directory(),
            f"{SIMDATA_FILENAME}.csv",
        )
        # Determine the usage of: 
        # - Each of the links in each of the time intervals
        cols = ["begin", "end", "link", "counts"]
        df2 = df1[cols].groupby(
            by=["begin", "end", "link"], 
            as_index=False,
        ).agg("sum")
        df2.to_csv(_file)

        # Save paths to simulation files
        dict_ = {
            "simdata": _file,
        }
        if self.simulation_files is None:
            raise ValueError("Corresponding odmat must be file missing!")
        elif isinstance(self.simulation_files, Dict):
            if len(self.simulation_files) == 0:
                raise ValueError("Corresponding odmat must be file missing!")
            else:
                self.simulation_files.update(dict_)
        else:
            raise ValueError("TODO")

        # - Each of the links in each of the time intervals by each origin-
        #   destination pair in each of the time intervals
        cols = ["begin", "end", "link", "from", "to", "depart", "counts"]
        df3 = df1[cols].groupby(
            by=["begin", "end", "link", "from", "to", "depart"],
            as_index=False,
        ).agg("sum")
        return df1, df2, df3

    def _get_vehRoute_file(self):
        last_step = str(int(self.simulator_options["last_step"]) - 1)
        vehRoute_file = os.path.join(
            self.file_io.output_directory(),
            last_step,
            f"{VEHROUTE_FILENAME}.veh.xml",
        )
        if os.path.exists(vehRoute_file):
            return vehRoute_file
        # Try to search for the correct directory
        else:
            dirs = os.listdir(self.file_io.output_directory())
            _dirs = []
            for dir in dirs:
                if ut.str_is_number(dir):
                    _dirs.append(ut.str_to_int(dir))
            max_value = np.max(_dirs)
            vehRoute_file = os.path.join(
                # self.file_io.root_output_directory,
                self.file_io.output_directory(),
                f"{max_value}", # TODO: Based on number of dua iterations
                f"{VEHROUTE_FILENAME}.veh.xml",
            )
            return vehRoute_file

    @ut.timing
    def _read_vehRoute_file(
        self,
        x_vec: pd.DataFrame,
        file_prefix: str = "",
    ) -> pd.DataFrame:
        vehRoute_file = self._get_vehRoute_file()
        root = ut.read_xml_file(xml_file=vehRoute_file)
        _dict_data = []
        # Define dictionary that should be filled in with values
        _dict_template: Dict[str, Any] = {
            "counts": None,   # Vehicle count on a link
            "from": None,     # Origin
            "to": None,       # Destination
            "depart":None,    # Defines departure interval for an OD pair
            "begin": None,    # Defines usage of link in an interval
            "end": None,      # Defines usage of link in an interval
            "link": None,     # Defines the link used
            "exitTime": None, # Redundent data used to locate time interval...
        }
        for child in root:
            if child.tag == "vehicle":
                assign_dict = _dict_template.copy()
                assign_dict["from"] = child.attrib["fromTaz"]
                assign_dict["to"] = child.attrib["toTaz"]
                # NOTE: Flow 'id_' specified in the 'trips.xml' file corresponds
                # to a vehicle in the '*.veh.xml' file  
                id_ = int(child.attrib["id"].split(".")[0])
                depart = float(x_vec.iloc[id_]["begin"])
                for begin, end in self.scenario_data.intervals:
                    if depart < end:
                        # Determine the departure time interval of a vehicle
                        assign_dict["depart"] = begin
                        break
                for subchild in child:
                    if subchild.tag == "route":
                        links = subchild.attrib["edges"].strip(" ").split(" ")
                        exitTimes = subchild.attrib["exitTimes"].strip(" ").split(" ")
                        if len(links) == len(exitTimes):
                            for i in range(len(links)):
                                # Make a copy for each edge a vehicle traverses
                                # starting from some origin and going to some 
                                # destination
                                assign_dict_copy = assign_dict.copy()
                                exitTime = float(exitTimes[i])
                                link = links[i]
                                # Skip the event if the vehicle exits the
                                # link outside the time period of analysis
                                if exitTime > self.scenario_data.intervals[-1][1]:
                                    break
                                for interval in self.scenario_data.intervals:
                                    if (exitTime >= interval[0] and exitTime < interval[1]) or \
                                        exitTime == self.scenario_data.intervals[-1][1]:
                                        # Get the start of the interval in which
                                        # the link was used
                                        assign_dict_copy["begin"] = interval[0]
                                        # Get the end of the interval in which
                                        # the link was used
                                        assign_dict_copy["end"] = interval[1]
                                        # Get the link identifier 
                                        assign_dict_copy["link"] = link
                                        assign_dict_copy["counts"] = 1.0
                                        assign_dict_copy["exitTime"] = exitTime
                                        _dict_data.append(assign_dict_copy)
                                        break
                        else:
                            error = f"A problem was encountered when trying to \
                                parse network edge exit times. The simulator \
                                output data is not consistent."
                            raise ValueError(error)
        _types = {
            "counts": float,
            "from": str, 
            "to": str,
            "begin": float,
            "end": float,
            "link": str,
            "depart": float,
        }
        df = pd.DataFrame(data=_dict_data, dtype=object).astype(dtype=_types)
        return df
    
    # TODO: Create a django model serializer instead
    def _to_json(self):
        if isinstance(self.simulation_files, Dict):
            content = {
                "simulation_files": self.simulation_files, 
                # TODO: Maybe also save the simulation options for the run?
                # "simulator_options": self.simulation_options,
                "wall_time": self.wall_time,
                "sysusr_time": self.sysusr_time,
            }
            return json.dumps(content, indent=4, sort_keys=True)
        else:
            return None

    def save(self, *args, **kwargs):
        return super().save(*args, **kwargs)


class SingleTestScenario:

    def __init__(
        self,
        simulator_options: Dict[str, Any],
        simulator_class,
        args_dict: Dict[str, Any],
        func: Callable,
        func_args: Dict[str, Any],
        network_directory: str,
        ) -> None:
        # Set simulator options
        simulator_options_handler = simulator_class.get_options_handler()
        self._simulator_options = simulator_options_handler(simulator_options)

        self._generate_scenario(
            args_dict=args_dict,
            func=func,
            func_args=func_args,
            network_directory=network_directory,
        )
        self._simulation_run = None

    def _check_vars(self) -> None:
        return None

    def _generate_scenario(
        self,
        args_dict: Dict[str, Any],
        func: Callable,
        func_args: Dict[str, Any],
        network_directory: str,
        ) -> None:
        _dict = {
            "output_directory_identifier": "",
        }
        file_io = FileIO.objects.create(**_dict)

        sumo_network = SumoNetwork()
        sumo_network._read_network_files(network_directory)
        sumo_network.save()

        _dict = {"network": sumo_network}
        args_dict.update(_dict)

        scenario_data = ScenarioData.objects.create(**args_dict)

        x_vec = func(scenario_data=scenario_data, **func_args)
        x_vec = ut.construct_x_vec(
            scenario_data=scenario_data,
            demand_vec=x_vec,
        )

        _dict = {
            "scenario_data": scenario_data,
            "file_io": file_io,
            "simulator_options": self._simulator_options.get_options(),
        }        
        obj = SumoMesoSimulationRun(**_dict)
        self._simulation_run = obj
        df1, df2, df3, dat = obj.run_simulator(
            x_vec=x_vec,
            parse_tripinfos=True,
        )
        self._create_generation_bounds(
            df=x_vec,
            scenario_data=scenario_data,
            simulation_run=obj,
        )
        self._create_seed_odmats(
            df=x_vec,
            simulation_run=obj,
        )
        json_file = os.path.join(
            self._simulation_run.file_io.root_output_directory,
            f"config.json",
        )
        with open(json_file, mode="w") as _file:
            self._to_json(file_object=_file)
        
    def _to_json(self, file_object = None) -> str:
        if self._simulation_run is not None:
            # TODO: Make sure data is not None else raise error
            contents = {
                "scenario_data": self._simulation_run.scenario_data._to_json(),
                "file_io": self._simulation_run.file_io._to_json(),
            }
            if file_object is not None:
                return json.dump(
                    contents,
                    file_object,
                    indent=4,
                    sort_keys=True,
                )
            else:
                return json.dumps(contents, indent=4, sort_keys=True)
        else:
            raise ValueError("TODO")

    def _create_generation_bounds(
        self,
        df: pd.DataFrame,
        scenario_data: ScenarioData,
        simulation_run, # TODO: Type
        ) -> None:
        gencon = {}
        for _id in scenario_data.network.taz["id"]:
            gencon[_id] = np.sum(df.loc[(df["from"] == _id)]["value"].values)
        # TODO: Make sure the file does not already exist!
        gencon_file = os.path.join(
            simulation_run.file_io.output_directory(),
            f"{GENCON_FILENAME}.csv",
        )
        pd.DataFrame(data=[gencon], dtype=object).to_csv(gencon_file)

    def _create_seed_odmat(
        self,
        df: pd.DataFrame,
        simulation_run,
        file_extension: str,
        ) -> None:
        SEEDMAT_FILENAME = ""
        seedmat_file = os.path.join(
            simulation_run.file_io.output_directory(),
            f"{file_extension}.csv",
        )
        df.to_csv(seedmat_file)

    def _create_seed_odmats(self, df: pd.DataFrame, simulation_run) -> None:
        #
        df_copy = df.copy()
        df_copy["value"] = df_copy["value"] * (
            0.7 + 0.3 * np.random.uniform(0, 1, df_copy["value"].shape[0])
        )
        self._create_seed_odmat(
            df=df_copy,
            simulation_run=simulation_run,
            file_extension="seedmat07",
        )
        #
        df_copy = df.copy()
        df_copy["value"] = df_copy["value"] * (
            0.8 + 0.3 * np.random.uniform(0, 1, df_copy["value"].shape[0])
        )
        self._create_seed_odmat(
            df=df_copy,
            simulation_run=simulation_run,
            file_extension="seedmat08",
        )
        #
        df_copy = df.copy()
        df_copy["value"] = df_copy["value"] * (
            0.9 + 0.3 * np.random.uniform(0, 1, df_copy["value"].shape[0])
        )
        self._create_seed_odmat(
            df=df_copy,
            simulation_run=simulation_run,
            file_extension="seedmat09",
        )


class BaseDataOrganizer:

    def __init__(
        self,
        scenario_data: ScenarioData,
        file_io: FileIO,
        scenario_options: Dict[str, Any],
        ) -> None:
        self.scenario_data = scenario_data
        self.file_io = file_io

        # Set scenario options
        self._scenario_options = BaseDataOrganizerOptions(
            options=scenario_options,
        )

        # Validate given input data
        self._check_vars()

        self._input_scenario_data: Dict[str, Any] = {}
        self._read_input_scenario_data(
            seedmat=self._scenario_options.get_value("seedmat"),
        )

        self.shape = self.get_shape()


    def _check_vars(self) -> None:
        if not isinstance(self.scenario_data, ScenarioData):
            raise ValueError("TODO")
        if not isinstance(self.file_io, FileIO):
            raise ValueError("TODO")
        return None


    def _read_input_scenario_data(self, seedmat: str) -> None:
        # Switch output <--> input directory
        if (self.file_io.root_input_directory == "" and \
            self.file_io.root_output_directory != ""):
            self.file_io.root_input_directory = os.path.join(
                self.file_io.root_output_directory,
                str(self.file_io.directory_counter)
            )
            _dict = {
                "output_directory_identifier": \
                    self.file_io.output_directory_identifier,
                "root_input_directory": self.file_io.root_input_directory,
            }
            self.file_io = FileIO(**_dict)
            self.file_io.save()
        else:
            raise ValueError("TODO")

        def _read_csv_into_dataframe(
            key: str,
            filename: str,
            index_col: int = 0,
            ) -> None:
            filepath = os.path.join(
                self.file_io.root_input_directory,
                filename,
            )
            if os.path.exists(filepath):
                self._input_scenario_data[key] = pd.read_csv(
                    filepath,
                    index_col=index_col,
                )
            else:
                raise ValueError("TODO")

        # Load all necesarry files
        _read_csv_into_dataframe("true_odmat", f"{ODMAT_FILENAME}.csv")
        _read_csv_into_dataframe("true_simdata", f"{SIMDATA_FILENAME}.csv")
        _read_csv_into_dataframe("gencon", f"{GENCON_FILENAME}.csv")
        _read_csv_into_dataframe("seed_odmat", f"{seedmat}.csv")

    @property
    def seed_odmat(self) -> pd.DataFrame:
        try:
            return self._input_scenario_data["seed_odmat"]
        except KeyError:
            return None
        
    @property
    def true_odmat(self) -> pd.DataFrame:
        try:
            return self._input_scenario_data["true_odmat"]
        except KeyError:
            return None

    @property
    def true_simdata(self) -> pd.DataFrame:
        try:
            return self._input_scenario_data["true_simdata"]
        except KeyError:
            return None

    @property
    def gencon(self) -> pd.DataFrame:
        try:
            return self._input_scenario_data["gencon"]
        except KeyError:
            return None

    def _set_lower_bounds(
        self,
        lower_bounds: Union[None, np.ndarray],
        ) -> np.ndarray:
        if lower_bounds is not None:
            return lower_bounds
        else: # Set defaults
            lower_bounds = np.zeros(self.shape, dtype=float)
            logging.info("TODO")
            return lower_bounds

    def _set_upper_bounds(
        self,
        upper_bounds: Union[None, np.ndarray],
        scale: float = 1.5,
        ) -> np.ndarray:
        if upper_bounds is not None:
            return upper_bounds
        else: # Set defaults
            if self.true_odmat is not None:
                values = self.true_odmat["value"].values
                upper_bounds = np.full(
                    self.shape,
                    float(scale * np.max(values)),
                    dtype=float,
                )
                logging.info("TODO")
            else:
                raise ValueError("TODO")
            return upper_bounds

    def get_shape(self) -> int:
        return self.scenario_data.network.total_od_pairs * \
            self.scenario_data.aggregation_intervals


class BaseTravelDemandEstimator(BaseDataOrganizer):

    def __init__(
        self,
        scenario_data: ScenarioData,
        file_io: FileIO,
        simulator_class,
        simulator_options: Dict[str, Any],
        estimator_class,
        estimator_options: Dict[str, Any],
        scenario_options: Dict[str, Any],
        collect_directory: str,
        lower_bounds=None,
        upper_bounds=None,
        ) -> None:
        super().__init__(
            scenario_data=scenario_data,
            file_io=file_io,
            scenario_options=scenario_options ,
        )

        self.simulator = simulator_class

        # Set output path for collecting the final output of an algorithm
        self.collect_directory = os.path.join(
            settings.MEDIA,
            collect_directory,
        )
        if not os.path.exists(self.collect_directory):
            os.makedirs(self.collect_directory)
        
        # Set estimator options
        options = {"x0_size": self.shape}
        options.update(estimator_options)
        if estimator_class is not None:
            estimator_options_handler = estimator_class.get_options_handler()
            self._estimator_options = estimator_options_handler(options)
        else:
            self._estimator_options = None

        # Set simulator options
        simulator_options_handler = simulator_class.get_options_handler()
        self._simulator_options = simulator_options_handler(simulator_options)

        # Collect necessary database model in a dict
        self.models: Union[None, Dict[str, Any]] = None
        # Class variable for saving the current estimation task object
        self.estimation_task: Union[None, EstimationTask] = None
        # Class variable for saving the current iteration object
        self.iteration_obj: Union[None, Iteration] = None
        # Class variable for saving the current objective function value object
        self.of_obj: Union[None, ObjectiveFunctionEvaluation] = None

        # Set lower and upper bounds
        self.lower_bounds = self._set_lower_bounds(lower_bounds=lower_bounds)
        self.upper_bounds = self._set_upper_bounds(upper_bounds=upper_bounds)

        # Validate input data
        # TODO: Check if it should also be called here. It is already being
        # called in the parent '__init__' method 
        self._check_vars()

    @abstractmethod
    def collect_final_output(self) -> None:
        raise NotImplementedError("TODO")

    def _check_vars(self) -> None:
        super()._check_vars()
        return None

    def apply_constraints(
        self,
        x_vec: pd.DataFrame,
        *args,
        **kwargs
        ) -> pd.DataFrame:
        # Apply lower bounds
        if self.lower_bounds is not None:
            x_vec["value"] = np.maximum(x_vec["value"], self.lower_bounds)
        # Apply upper bounds
        if self.upper_bounds is not None:
            x_vec["value"] = np.minimum(x_vec["value"], self.upper_bounds)
        return x_vec

    def termination_criteria(self) -> bool:
        """A method that determines when we should terminate an algorithm run.

        Note: This method should return True or False.
              By default we use:
                1. The maximum number of algorithm iterations (hard criterion)
                2. The maximum number of objective function evaluations \
                    (soft criterion) as the termination criteria.

        Returns:
            (bool): True is returned if the termination has been reached, else \
                False.
        """
        # Fetch the stopping criteria which should be used
        stopping_criteria = self._estimator_options.get_value(
            "stopping_criteria"
        )
        # Fetch the current values
        max_iters = self._estimator_options.get_value("max_iters")
        max_evals = self._estimator_options.get_value("max_evals")
        if self.estimation_task is not None:
            iter_counter = self.estimation_task.iter_counter
            eval_counter = self.estimation_task.eval_counter
            if "max_iters" in stopping_criteria:
                if iter_counter > max_iters:
                    print(
                        "INFO    : \
                            ALGORITHM TERMINATED. Max algorithm iterations reached!"
                    )
                    print(
                        f"INFO    : {iter_counter} (iteration) > {max_iters} \
                            (max iterations) --> {iter_counter > max_iters}"
                    )
                    return True
            if "max_evals" in stopping_criteria:
                if eval_counter > max_evals:
                    print(
                        "ALGORITHM TERMINATED: \
                            Max objective function evaluations reached!"
                    )
                    print(
                        f"OF Evaluations > Max OF Evaluations: {eval_counter} > \
                            {max_evals}: {eval_counter > max_evals}"
                    )
                    return True
            return False
        else:
            raise ValueError("TODO")

    def get_estimator_options(self) -> Dict[str, Any]:
        """Retrieve the estimator options that have been set.

        Returns:
            Dict[str, Any]: A dictionary containing all set and used \
                optimization settings.
        """
        if self._estimator_options is not None:
            return self._estimator_options.__options
        else:
            raise ValueError("TODO")

    @abstractmethod
    def run(
        self,
        x_vec: pd.DataFrame,
        collect_final_output: bool = False,
        ) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
        """Main method call for an estimator object."""
        raise NotImplementedError("TODO")
    
    def _sum_of_values(self, of_values: Dict[str, Tuple[float, float]]) -> float:
        return np.sum(
            [weight * of_value for weight, of_value in of_values.values()]
        )

    def objective_function(
        self,
        x_vec: pd.DataFrame,
        ) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame, pd.DataFrame]:
        if self.models is not None and self.estimation_task is not None:
            start_sysusr = time.process_time()
            start_wall = time.time()
            of_values, simdata, assignmat, simulator_object = \
                self._objective_function(x_vec=x_vec)
            end_sysusr = time.process_time()
            end_wall = time.time()

            # Update the objective function evaluation counter
            dict_ = {"eval_counter": self.estimation_task.eval_counter + 1}
            self.models["EstimationTask"].objects.filter(
                uid=self.estimation_task.uid
            ).update(**dict_)
            self.estimation_task = self.models["EstimationTask"].objects.get(
                uid=str(self.estimation_task.uid)
            )       
            dict_ = {
                "estimation_task": self.estimation_task,
                "iteration": self.iteration_obj,
                "of": of_values,
                # Save the (wall) time it took to evaluate the objective function.
                "wall_time": end_wall - start_wall,
                # Save the (system + user) time it took to evaluate the objective 
                # function.
                "sysusr_time": end_sysusr - start_sysusr,
                "simulation_data": simulator_object._to_json(),
            }
            self.of_obj = self.models["ObjectiveFunctionEvaluation"].objects.create(
                **dict_
            )
            return of_values, simdata, assignmat
        else:
            raise ValueError("TODO")

    def _objective_function(
        self,
        x_vec: pd.DataFrame,
        ) -> Tuple[
            Dict[str, Tuple[float, float]], pd.DataFrame, pd.DataFrame, Any,
        ]:
        _dict = {
            "scenario_data": self.scenario_data,
            "file_io": self.file_io,
            "simulator_options": self._simulator_options.get_options(),
        }
        simulator_object = self.simulator(**_dict) # TODO: Save object to db?
        _, simdata, assignmat, _ = simulator_object.run_simulator(x_vec=x_vec)
        # Associate the serialized simulator object with an objective function
        # value object
        of_values = self.evaluate_of_terms(
            x_vec=x_vec,
            simdata=simdata,
            assignmat=assignmat,
        )
        return of_values, simdata, assignmat, simulator_object

    @abstractmethod
    def evaluate_of_terms(
        self,
        x_vec: pd.DataFrame,
        simdata: pd.DataFrame,
        assignmat: pd.DataFrame,
        ) -> Dict[str, Tuple[float, float]]:
        raise NotImplementedError("TODO")

    def evaluate_odmat(
        self,
        odmat: pd.DataFrame
        ) -> Dict[str, Tuple[float, float]]:
        # Retrieve the corresponding weights given as an estimator option
        weight: Dict[str, float] = self._estimator_options.get_value(
            "weight_configuration"
        )
        # Compute OF value if seed odmat is available
        if self.seed_odmat is not None:
            df_odmat = self.seed_odmat.merge(
                odmat,
                on=["begin", "end", "from", "to"],
                how="outer",
            )
            # NOTE: 'value_x' will be the given, constant, input seed demands
            # NOTE: 'value_y' will be the estimated demands
            val_odmat = np.linalg.norm(
                x=df_odmat["value_y"].sub(df_odmat["value_x"]),
                ord=2,
            ) / np.linalg.norm(
                x=df_odmat["value_x"],
                ord=2,
            )
            return {
                "odmat": (weight["odmat"], val_odmat),
            }
        else:
            return {
                "odmat": (weight["odmat"], 0.0),
            }

    def evaluate_counts(
        self,
        merged_simdata: pd.DataFrame
        ) -> Dict[str, Tuple[float, float]]:
        # Retrieve the corresponding weights given as an estimator option
        weight: Dict[str, float] = self._estimator_options.get_value(
            "weight_configuration"
        )

        # TODO: Make sure 'counts_x' and 'counts_y' are present in the
        # dataframe and that 'self.of_weight_config['counts']' is present

        # NOTE: 'counts_x' will be the given, constant, input counts
        # NOTE: 'counts_y' will be the counts from a simulation resulting
        #       from the estimated demands
        val_counts = np.linalg.norm(
            x=merged_simdata["counts_y"].sub(merged_simdata["counts_x"]),
            ord=2,
        ) / np.linalg.norm(
            x=merged_simdata["counts_x"],
            ord=2,
        )
        return {
            "counts": (weight["counts"], val_counts),
        }

    def evaluate_simdata(
        self,
        simdata: pd.DataFrame,
        ) -> Dict[str, Tuple[float, float]]:
        if self.true_simdata is not None:
            merged_simdata = self.true_simdata.merge(
                simdata,
                on=["link", "begin", "end"],
                how="outer",
            )
            # Locate links equipped with sensors. We should only determine
            # the discrepancy between observed and simulated quantities on
            # links equipped with sensors
            merged_simdata = merged_simdata.loc[
                merged_simdata["link"].isin(
                    self.scenario_data.network.det_subset["link"],
                )
            ]
            
            # If the simulation output data produced for a certain input 'x_vec' 
            # specifying trips between origins and destionations do not produce 
            # data on certain links, then set these values to 0 by default
            merged_simdata = merged_simdata.fillna(0)

            # Calculate difference to observed counts
            counts_val = self.evaluate_counts(merged_simdata=merged_simdata)
            return counts_val
        else:
            raise ValueError("TODO")

 
class GradientBasedTravelDemandEstimator(BaseTravelDemandEstimator):

    # def find_max_step_length(
    #     self,
    #     x_vec: pd.DataFrame,
    #     ghat: pd.Series,
    #     step_inc: float = 1e18,
    #     step_inc_lower_bound: float = 1e-18,
    #     ) -> float:
    #     # Initial step length in the descent direction
    #     max_step_length = 0.0
    #     x_vec_copy = x_vec.copy()
    #     # Loop until generation constraints are binding
    #     while True:
    #         # Initially we do not violate any lower/uppder bound constraints
    #         violBoundsCon = False
    #         # Initially we don't violate any generation constraints
    #         violGenCon = False
    #         max_step_length += step_inc
    #         x_vec_copy["value"] = x_vec_copy["value"] - max_step_length * ghat
    #         # x_vec_copy["value"] = x_vec["value"] - max_step_length * ghat
    #         # Only apply lower bound constraints to not constrain the search  
    #         x_vec_copy = self.apply_constraints(x_vec=x_vec_copy)
    #         # Determine if all constrains are bidning
    #         consBinding = all(
    #             (x_vec_copy["value"] == self.lower_bounds) | \
    #             (x_vec_copy["value"] == self.upper_bounds)
    #         )

    #         # x_vec_copy["value"] = np.maximum(
    #         #     x_vec_copy["value"],
    #         #     self.lower_bounds,
    #         # )

    #         # if violBoundsCon is False:
    #         #     for row_name in self.gencon:
    #         #         values = x_vec_copy.loc[
    #         #             (x_vec_copy["from"] == row_name),
    #         #             "value",
    #         #         ].to_frame(name="value")
    #         #         _sum = values["value"].sum()

    #                 # if _sum > self.gencon[row_name].iat[0] or _sum <= 0:
    #                 #     violGenCon = True
    #                 #     break

    #         if violBoundsCon is False:
    #             for row_name in self.gencon:
    #                 values = x_vec_copy.loc[
    #                     (x_vec_copy["from"] == row_name),
    #                     "value",
    #                 ].to_frame(name="value")
    #                 _sum = values["value"].sum()
    #                 print("_sum", _sum, "self.gencon[row_name].iat[0]" , self.gencon[row_name].iat[0], _sum > self.gencon[row_name].iat[0])
    #                 if _sum > self.gencon[row_name].iat[0]:
    #                     violGenCon = True
    #                     break
    #         # if violGenCon or violBoundsCon:
    #         if violGenCon or consBinding:
    #             # Reset variables
    #             max_step_length -= step_inc
    #             x_vec_copy = x_vec.copy()
    #             # Halve the step increment and try again...
    #             step_inc /= 2.0
    #             # Return the max_step_length, when we've gotten sufficiently
    #             # close to the maximum possible step length in the descent 
    #             # direction...
    #             if step_inc < step_inc_lower_bound:
    #                 print(f"Setting new max step-length to: {max_step_length}")
    #                 quit()
    #                 return max_step_length


    def find_max_step_length(
        self,
        x_vec: pd.DataFrame,
        ghat: pd.Series,
        step_inc: float = 1e8,
        step_inc_lower_bound: float = 1e-8,
        ) -> float:
        # Initial step length in the descent direction
        max_step_length = 0.0
        previous_max_step_length = 0.0
        x_vec_copy = x_vec.copy()
        # Loop until generation constraints are binding
        while True:
            violGenCon = False
            max_step_length += step_inc
            x_vec_copy["value"] = x_vec["value"] - max_step_length * ghat
            # Only apply lower bound constraints to not constrain the search  
            x_vec_copy = self.apply_constraints(x_vec=x_vec_copy)

            # Determine if all upper or lower bound constraints are binding
            consBinding = all(
                (x_vec_copy["value"] == self.lower_bounds) | \
                (x_vec_copy["value"] == self.upper_bounds)
            )

            # If upper/lower bound constraints are not binding then check if 
            # generation constraints are violated
            if consBinding is False:
                for row_name in self.gencon:
                    values = x_vec_copy.loc[
                        (x_vec_copy["from"] == row_name),
                        "value",
                    ].to_frame(name="value")
                    _sum = values["value"].sum()
                    if _sum > self.gencon[row_name].iat[0]:
                        violGenCon = True
                        break
            # If upper/lower bound constraints are binding or generation 
            # constraints are violated then backtrack and try again with a 
            # smaller step size
            if violGenCon or consBinding:
                previous_max_step_length = max_step_length
                # Backtrack
                max_step_length -= step_inc
                # Halve the step increment and try again...
                step_inc /= 2.0

                # Return the max_step_length, when we've gotten sufficiently
                # close to the maximum possible step length in the descent 
                # direction...
                # if (step_inc < step_inc_lower_bound) or \
                #     (previous_max_step_length == max_step_length):
                #     print(f"Setting new max step-length to: {max_step_length}")
                #     return max_step_length
            if step_inc < step_inc_lower_bound:
                print()
                print("    ::::Lower bound on step-length reached: ", step_inc, previous_max_step_length, max_step_length)
                print(f"Setting new max step-length to: {max_step_length}")
                print()
                return max_step_length
            # print(np.abs(previous_max_step_length - max_step_length), step_inc)
            # else:
            #     if (previous_max_step_length == max_step_length):
            #         print()
            #         print("    ::::Converged to a max step-length: ", step_inc, previous_max_step_length, max_step_length)
            #         print(f"Setting new max step-length to: {max_step_length}")
            #         print()
            #         return max_step_length
                # print(
                #     "previous: ", previous_max_step_length,
                #     "next    : ", max_step_length,
                # )
                # previous_max_step_length = max_step_length
            
    def f_d0(self, x, coefs) -> float:
        """
        The 0. order derivative of a 2nd degree polynomial in 1 variable.
        """
        return coefs[2] * x * x + coefs[1] * x + coefs[0]

    def f_d1(self, x, coefs) -> float:
        """
        The 1. order derivative of a 2nd degree polynomial in 1 variable.
        """
        return 2 * coefs[2] * x + coefs[1]

    def f_d2(self, x, coefs) -> float:
        """
        The 2. order derivative of a 3rd degree polynomial in 1 variable.
        """
        return 2 * coefs[2]

    def find_min(self, coefs, a: float, b: float) -> float:
        alpha = -coefs[1] / (2 * coefs[2])
        f_alpha = self.f_d0(alpha, coefs)
        print("Local min alpha   : ", alpha)
        print("Local min f(alpha): ", f_alpha)
        # Check endpoints
        if self.f_d0(a, coefs) < f_alpha or alpha < 0:
            alpha = a
            f_alpha = self.f_d0(a, coefs)
            print("alpha ", alpha, " f_alpha ", f_alpha, " a ", a)
        elif self.f_d0(b, coefs) < f_alpha or alpha > b:
            alpha = b
            f_alpha = self.f_d0(b, coefs)
            print("alpha ", alpha, " f_alpha ", f_alpha, " b ", b)
        return alpha

    @abstractmethod
    def _generate_poly_data(
        self,
        x_vec: pd.DataFrame,
        of_values: Dict[str, Tuple[float, float]],
        ghat: pd.Series,
        max_step_length: float,
        extra_data: Dict[str, Any]
        ) -> Dict[str, Any]:
        raise NotImplementedError("TODO")

    def _polynomial_regression(
        self,
        x_vec: pd.DataFrame,
        of_values: Dict[str, Tuple[float, float]],
        ghat: pd.Series,
        extra_data: Dict[str, Any],
        ) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]], Dict[str, Any]]:
        # Decide the upper_bound on the step-length
        max_step_length = self.find_max_step_length(x_vec=x_vec, ghat=ghat)
        if max_step_length < 1e-18:
            logging.info(
                f"The step length is too small: {max_step_length}. Returning \
                original data."
            )
            # TODO: Re-evaluate otherwise we get stuck in an endless loop, as 
            # a minimum as been found 
            extra_data.update({
                    "reevaluate": True,
            })
            return x_vec, of_values, extra_data
        else:
            poly_data = self._generate_poly_data(
                x_vec=x_vec,
                of_values=of_values,
                ghat=ghat,
                max_step_length=max_step_length,
                extra_data=extra_data,
            )
            # Extract required data
            x_vals = poly_data["x_vals"]
            of_vals = poly_data["of_vals"]
            simdata_vals = poly_data["simdata_vals"]
            step_lengths = poly_data["step_lengths"]
            # NOTE: If 'assignmat_vals' is contained in the dictionary then the
            # 'AssignmentMatrixBasedTravelDemandEstimator' is used. Otherwise,
            # another estimator is used.
            reevaluate = True
            try:
                assignmat_vals = poly_data["assignmat_vals"]
            except KeyError:
                assignmat_vals = None
            # Fit a polynomial of a certain degree to the data, which was just
            # collected in the for-loop above, i.e., data: 
            # [(x_k1, of_k1),...,(x_kn, of_kn)]
            ys = [self._sum_of_values(of_values=vals) for vals in of_vals]
            coefs = poly.polyfit(
                x=step_lengths,
                y=ys,
                deg=2,
            )
            # Get the current smallest objective function value and its 
            # corresponding index in the list objective function values: 
            # "of_vals"
            # local_min = self.find_min(coefs, 0, np.max(step_lengths))
            local_min = self.find_min(coefs, 0, max_step_length)
            # if local_min > 0 and local_min < max(step_lengths):
            if local_min > 0 and local_min < max_step_length:
                print(
                    f"Found an objective function minimum using step-length: \
                    {local_min}"
                )
                x_new = x_vec.copy()
                print(
                    f"Evaluating an additional point using the step-length: \
                    {local_min}"
                )
                x_new["value"] = x_vec["value"] - local_min * ghat
                of_values_new, simdata_new, assignmat_new = \
                    self.objective_function(x_vec=x_new)
                of_vals.append(of_values_new)
                x_vals.append(x_new)
                step_lengths.append(local_min)
                simdata_vals.append(simdata_new)
                if assignmat_vals is not None:
                    assignmat_vals.append(assignmat_new)
                    reevaluate = False
            # If there is no change in the trip demands then simply return the
            # current best objective function value and corresponding dataframe
            # of trip demands
            ys = [self._sum_of_values(of_values=vals) for vals in of_vals]
            of_min_index, of_min_value = min(
                enumerate(ys),
                key = lambda v: v[1],
            )
            print(
                f"Using step-length: {step_lengths[of_min_index]}, which \
                produced a minimum objective function value."
            )
            # TODO: Make the following a commandline arg or input option
            plot_poly = True
            if plot_poly is True:
                self._plot_polynomial(coefs, step_lengths, ys, local_min)

            # NOTE: If 'assignmat_vals' is not None then the
            # 'AssignmentMatrixBasedTravelDemandEstimator' is used. Otherwise,
            # another estimator is used and only 'simdata' should be returned
            if assignmat_vals is not None:
                extra_data.update({
                    "assignmat": assignmat_vals[of_min_index],
                    "reevaluate": reevaluate,
                })
            else:
                extra_data = {"simdata": simdata_vals[of_min_index]}
            return x_vals[of_min_index], of_vals[of_min_index], extra_data

    def _plot_polynomial(
        self,
        coefs,
        step_lengths: List[float],
        ys: List[float],
        root,
    ) -> None:
        """
        At each iteration, plot the polynomial which was fitted to the data.
        """
        if self.estimation_task is not None:
            plt.title(
                "Iteration {:04.0f}".format(self.estimation_task.iter_counter)
            )
            plt.xlabel("Step Length"); plt.ylabel("Objective Function Value")
            plt.scatter(step_lengths, ys, marker = "x", label = "Sample points")
            inc = np.max(step_lengths) / 250
            x = [i * inc for i in range(250 + 1)]
            of = [self.f_d0(i, coefs) for i in x]
            plt.plot(x, of, "--", label = "Fitted polynomial")
            if root is not None:
                plt.scatter(
                    root,
                    self.f_d0(root, coefs),
                    color="red",
                    label="Minimum",
                )
            plt.legend(); plt.grid(True)
            # Save figure to log data directory
            plt.savefig(
                os.path.join(
                    self.file_io.root_output_directory,
                    "poly_{:04.0f}.png".format(self.estimation_task.iter_counter),
                )
            )
            plt.close()
        else:
            raise ValueError("TODO")


class SPSABasedTravelDemandEstimator(GradientBasedTravelDemandEstimator):

    def __init__(
        self,
        scenario_data: ScenarioData,
        file_io: FileIO,
        scenario_options: Dict[str, Any],
        simulator_class,
        simulator_options: Dict[str, Any],
        estimator_options: Dict[str, Any],
        collect_directory: str,
        lower_bounds=None,
        upper_bounds=None
        ) -> None:
        super().__init__(
            scenario_data=scenario_data,
            file_io=file_io,
            scenario_options=scenario_options,
            simulator_class=simulator_class,
            simulator_options=simulator_options,
            estimator_class=self,
            estimator_options=estimator_options,
            collect_directory=collect_directory,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        self.models = {
            EstimationTaskMetaData.__name__: EstimationTaskMetaData,
            EstimationTask.__name__: EstimationTask,
            Iteration.__name__: Iteration,
            ObjectiveFunctionEvaluation.__name__: ObjectiveFunctionEvaluation,
        }
        metadata = self.models["EstimationTaskMetaData"].objects.create(
            **{
                "file_io": file_io,
                "scenario_data": scenario_data,                
            },
        )
        self.estimation_task = self.models["EstimationTask"].objects.create(
            **{
                "metadata": metadata,
            },
        )

    @staticmethod
    def get_options_handler():
        return SPSAEstimatorOptions

    def create_perturbation(
        self,
        x_vec: pd.DataFrame,
        c_k: float,
        delta: np.ndarray,
        sign: str,
    ) -> pd.DataFrame:
        """Perturb the current approximate solution vector.

        Args:
            x (numpy.ndarray): The current approximate solution.
            c_k (numpy.float64): The perturbation parameter at iteration k. This \
                parameter is used to scale the perturbation vector.
            delta (numpy.ndarray): The perturbation vector i.e., a vector of elements \
                that are used to perturb the elements of the current approximate \
                solution s.t. we can eventually approximate the gradient.
            sign (str): An option that specifies whether the perturbation should be \
                negative or positive.

        Raises:
            ValueError: If the wrong string input is provided.

        Returns:
            numpy.ndarray: The perturbed approximate solution vector.
        """
        x_new = x_vec.copy()
        if sign.lower() == "plus":
            x_new["value"] = x_vec["value"] + c_k * delta
            x_new = self.apply_constraints(x_vec=x_new)
        elif sign.lower() == "minus":
            x_new["value"] = x_vec["value"] - c_k * delta
            x_new = self.apply_constraints(x_vec=x_new)
        else:
            # Internal error.
            raise ValueError(
                "Perturbation sign wrongly specified. "
                + "'sign' needs to either be 'plus' or 'minus'"
            )
        return x_new

    def _initialize(
        self,
        x_vec: pd.DataFrame
        ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
        """Evaluate the starting point/initial guess "x0", without setting a \
            descent direction. A descent direction is set for all other \
            iterations.
        """
        if self.models is not None and self.estimation_task is not None:
            # Create and save a new (approximate) solution object
            dict_ = {
                "estimation_task": self.estimation_task,
            }
            self.iteration_obj = self.models["Iteration"].objects.create(**dict_)
            # Update the objective function evaluation counter
            dict_ = {"eval_counter": self.estimation_task.eval_counter}
            self.models["EstimationTask"].objects.filter(
                uid=self.estimation_task.uid
            ).update(**dict_)
            self.estimation_task = self.models["EstimationTask"].objects.get(
                uid=str(self.estimation_task.uid)
            )       
            # Evaluate the starting point/initial guess
            of_values, simdata, _ = self.objective_function(x_vec=x_vec)
            return of_values, {
                "simdata": simdata,
            }
        else:
            raise ValueError("TODO")

    def update_spsa_params(self) -> Tuple[float, float]:
        """Update the iteration-dependent parameters that are used in the \
            basic/vanilla SPSA algorithm. The parameters are c_k and a_k. These \
            decrease monotonically in each  iteration. Appropriate initial parameter \
            values for parameters c and a should be determined through experimentation.

        Returns:
            Tuple[numpy.float64, numpy.float64]: The step-length in the descent \
                direction at iteration k and the perturbation parameter at iteration \
                k. The perturbation parameter parameter is used to scale the \
                perturbation vector.
        """
        if self.estimation_task is not None and \
            self._estimator_options is not None:
            k = self.estimation_task.iter_counter
            c = self._estimator_options.get_value("param_c")
            a = self._estimator_options.get_value("param_a")
            A = self._estimator_options.get_value("param_A")
            gamma = self._estimator_options.get_value("param_gamma")
            alpha = self._estimator_options.get_value("param_alpha")
            c_k = c / ((k + 1) ** gamma)
            a_k = a / (((k + 1) + A) ** alpha)
            return a_k, c_k
        else:
            raise ValueError("TODO")
    
    def collect_final_output(
        self,
        x_vec: pd.DataFrame,
        extra_data: Dict[str, Any],
        ) -> None:
        if (self.models is not None) and \
            (self.estimation_task is not None) and \
            (self._estimator_options is not None):
            # Fecth latest estimation task db object
            estimation_task = self.models["EstimationTask"].objects.get(
                uid=str(self.estimation_task.uid)
            )
            weight: Dict[str, float] = self._estimator_options.get_value(
                "weight_configuration"
            )
            try:
                if weight["odmat"] <= 0.0:
                    seedmat = None
                else:
                    seedmat = self._scenario_options.get_value("seedmat")
            except KeyError:
                seedmat = None
            simdata = extra_data["simdata"]
            merged_simdata = self.true_simdata.merge(
                simdata,
                on=["link", "begin", "end"],
                how="outer",
            )
            merged_odmat = self.true_odmat.merge(
                x_vec,
                on=["begin", "end", "from", "to"],
                how="outer",
            )
            output_filename = f"gradient-spsa_{seedmat}"
            output = {
                "experiment_metadata": {
                    "method": {
                        "estimator": "assignmat",
                        "estimator_options": self._estimator_options.get_options(),
                    },
                    "seedmatrix": seedmat,
                },
                # Estimated and observed demands
                "merged_odmat": merged_odmat.to_json(),
                # Estimated and oberved quantities
                "merged_simdata": merged_simdata.to_json(),
                # The history of objective function values
                "of_history": [
                    of_eval.of for of_eval in estimation_task.of_evals.all()
                ],
                # Add the computational time used in each step of the applied
                # estimation method
                "timing": {
                    # Save the (wall) time 
                    "wall_time": [extra_data["wall_time"]],
                    # Save the (system + user) time 
                    "sysusr_time": [extra_data["sysusr_time"]],
                },
            }
            output_filepath = os.path.join(
                self.collect_directory,
                f"{output_filename}.json",
            )
            with open(output_filepath, mode="w") as file_object:
                json.dump(output, file_object, indent=4, sort_keys=True)
        else:
            raise ValueError("TODO")

    def run(
        self,
        x_vec: pd.DataFrame,
        collect_final_output: bool = False,
        ) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
        start_sysusr = time.process_time()
        start_wall = time.time()
        if self._estimator_options is not None:
            # Set the given random seed for reproducibility purposes
            np.random.seed(
                self._estimator_options.get_value("random_seed")
            )
            ghat_magnitude_atol = self._estimator_options.get_value(
                "ghat_magnitude_atol"
            )
            if self.models is not None and self.estimation_task is not None:
                of_values, extra_data = self._initialize(x_vec=x_vec)
                # Enter the main algorithm loop
                while self.termination_criteria() is False:
                    # Update the objective function evaluation counter
                    dict_ = {
                        "iter_counter": self.estimation_task.iter_counter + 1
                    }
                    self.models["EstimationTask"].objects.filter(
                        uid=self.estimation_task.uid
                    ).update(**dict_)
                    self.estimation_task = self.models["EstimationTask"].objects.get(
                        uid=str(self.estimation_task.uid)
                    )  
                    # Create and save a new iteration object
                    dict_ = {
                        "estimation_task": self.estimation_task,
                    }
                    self.iteration_obj = self.models["Iteration"].objects.create(
                        **dict_,
                    )
                    # Apply possible constraints and update the current approximate 
                    # solution
                    x_vec = self.apply_constraints(x_vec=x_vec)
                    # Update SPSA parameters
                    a_k, c_k = self.update_spsa_params()
                    # Approximate the gradient
                    ghat, magnitude = self.approximate_gradient(
                        x_vec=x_vec,
                        extra_data={
                            "of_values": of_values,
                            # NOTE: Fix perturbation parameter to 1 to be 
                            #       compatible with SUMO
                            "c_k": 1.0,
                            "a_k": a_k,
                        },
                    )
                    # Check if we have arrived at a local minimum
                    if not np.isclose(magnitude, 0, atol=ghat_magnitude_atol, rtol=0):
                        # Take a step in the descent direction
                        extra_data.update({"a_k": a_k})
                        x_vec, of_values, extra_data = self.take_step(
                            x_vec=x_vec,
                            of_values=of_values,
                            ghat=ghat,
                            extra_data=extra_data,
                        )
                    else:
                        break
                end_sysusr = time.process_time()
                end_wall = time.time()
                if collect_final_output:
                    extra_data.update({
                        # Save the (wall) time for the complete algorithm run
                        "wall_time": end_wall - start_wall,
                        # Save the (system + user) time for the complete 
                        # algorithm run
                        "sysusr_time": end_sysusr - start_sysusr,
                    })
                    self.collect_final_output(
                        x_vec=x_vec,
                        extra_data=extra_data,
                    )
                return of_values, x_vec
            else:
                raise ValueError("TODO")
        else:
            raise ValueError("TODO")

    def take_step(
        self,
        x_vec: pd.DataFrame,
        of_values: Dict[str, Tuple[float, float]],
        ghat: pd.Series,
        extra_data: Dict[str, Any],
        ) -> Tuple[
            pd.DataFrame, Dict[str, Tuple[float, float]], Dict[str, Any]
        ]:
        # Adjust the current approximate solution and evaluate it
        x_final, of_final, extra_data = self._polynomial_regression(
                x_vec=x_vec,
                of_values=of_values,
                ghat=ghat,
                extra_data=extra_data,
        )
        return x_final, of_final, extra_data

    # def take_step(
    #     self,
    #     x_vec: pd.DataFrame,
    #     of_values, #Dict[str, numpy.float64],
    #     ghat: pd.Series,
    #     extra_data: Dict,
    #     ):
    #     a_k = extra_data["a_k"]
    #     d = -ghat
    #     # Adjust the current approximate solution and evaluate it
    #     x_vec["value"] = x_vec["value"] + a_k * d
    #     x_vec = self.apply_constraints(x_vec=x_vec)

    #     start_sysusr = time.process_time()
    #     start_wall = time.time()
    #     of_values, simdata, _ = self.objective_function(x_vec=x_vec)
    #     end_sysusr = time.process_time()
    #     end_wall = time.time()
    #     # # Determine whether the result should be saved or not
    #     # bool_statement = "all" in self._optimizer_options.get_value(
    #     #     "save_results"
    #     # ) or "iteration" in self._optimizer_options.get_value("save_results")
    #     # save = True if bool_statement else False
    #     # # Create and save a new objective function evaluation object
    #     dict_ = {
    #         "estimation_task": self.estimation_task,
    #         "iteration": self.iteration_obj,
    #         # Use a custom numpy file interface to interface with the peewee database
    #         # models and be able to read/write numpy arrays to the disk
    #         # "x": NumpyFile(
    #         #     # Give the current approximate solution as input
    #         #     array=self.x,
    #         #     # Depending on the value of "save" save the vector to the disk
    #         #     save=save,
    #         #     # Set the main output directory
    #         #     storage_path=self._optimizer_options.get_value("output_dir"),
    #         #     # Determine the file path and name of the numpy file containing the
    #         #     # current approximate solution that is to be saved within the main
    #         #     # output directory
    #         #     filename=enumerate_obj_iter_eval(
    #         #         optimization_task=self.optimization_task,
    #         #         obj_name="x",
    #         #     ),
    #         # ),
    #         "of": of_values,
    #         # Save the (wall) time it took to evaluate the objective function.
    #         "wall_time": end_wall - start_wall,
    #         # Save the (system + user) time it took to evaluate the objective 
    #         # function.
    #         "sysusr_time": end_sysusr - start_sysusr,
    #         "take_step": True,
    #     }
    #     self.of_obj = self.models["ObjectiveFunctionEvaluation"].objects.create(
    #         **dict_
    #     )
    #     return x_vec, of_values, {
    #         "simdata": simdata,
    #     }

    def evaluate_of_terms(
        self,
        x_vec: pd.DataFrame,
        simdata: pd.DataFrame,
        assignmat: pd.DataFrame # TODO: Needed?
        ) -> Dict[str, Tuple[float, float]]:
        return_dict = {}
        val_odmat = self.evaluate_odmat(odmat=x_vec)
        return_dict.update(val_odmat)
        val_simdata = self.evaluate_simdata(simdata=simdata)
        if val_simdata is not None:
            return_dict.update(val_simdata)

        # NOTE: TEMP. Merge 'self.seed_odmat' and 'x_vec' here.
        #       Merged dataframe is used in both 'evaluate_odmat' and 
        #       'odmat_deriv' 
        # self.odmat_derivative(odmat=x_vec)
        # self.simdata_derivative(
        #     odmat=x_vec,
        #     assignmat=assignmat,
        #     simdata=simdata
        # )
        print(":::: VAL_ODMAT  : ", val_odmat)
        print(":::: VAL_SIMDATA: ", val_simdata)

        return return_dict

    # def approximate_gradient(
    #     self,
    #     x_vec: pd.DataFrame,
    #     extra_data: Dict[str, Any],
    # ) -> Tuple[np.ndarray, float]:
    #     """Approximate the gradient through one-sided or two-sided finite difference \
    #         approximations.

    #     Args:
    #         x (numpy.ndarray): The current approximate solution.
    #         of (Dict[str, Any]): The corresponding objective function value(s).
    #         c_k (numpy.float64): The perturbation parameter at iteration k. This \
    #             parameter is used to scale the perturbation vector.

    #     Returns:
    #         Tuple[numpy.ndarray, numpy.float64]: The approximated gradient along with \
    #             its magnitude.
    #     """
    #     of = extra_data["of_values"]
    #     c_k = extra_data["c_k"]
    #     # Fetch some required and optional settings
    #     gradient_design = self._estimator_options.get_value("gradient_design")
    #     perturbation_type = self._estimator_options.get_value(
    #         "perturbation_type"
    #     )
    #     gradient_replications = self._estimator_options.get_value(
    #         "gradient_replications"
    #     )
    #     vec_size = self._estimator_options.get_value("x0_size")
    #     # Instantiate a vector that is going to hold the gradient estimate
    #     ghat_terms = np.zeros(
    #         shape=(vec_size, 1),
    #         dtype=float,
    #     )
    #     # Estimate the gradient at the current approximate solution:
    #     for i in range(gradient_replications):
    #         # Perturbation vector type:
    #         #   1. Random perturbations
    #         if perturbation_type == 1:
    #             delta = self.bernoulli_perturbation()
    #         #   2. Deterministic perturbations
    #         elif perturbation_type == 2:
    #             delta = self.hadamard_perturbation()
    #         #   3. Coordinate perturbations (equivalent to using the FDSA algorithm)
    #         elif perturbation_type == 3:
    #             # Generate the i'th unit vector
    #             delta = self.coordinate_perturbation(int(i))
    #         else:
    #             raise ValueError("TODO")
    #         # Approximate the gradient using a one-sided finite difference approximation
    #         if gradient_design == 1:
    #             of_m = self.gradient_design_1(x_vec, c_k, delta)
    #         # Approximate the gradient using a two-sided finite difference approximation
    #         elif gradient_design == 2:
    #             of_m, of_p = self.gradient_design_2(x_vec, c_k, delta)
    #         else:
    #             raise ValueError("TODO")
    #         # Approximate the gradient
    #         if gradient_design == 1 and perturbation_type in [1, 2]:
    #             _of_m = self._sum_of_values(of_values=of_m)
    #             _of = self._sum_of_values(of_values=of)
    #             ghat_terms[:, 0] += (_of - _of_m) / (c_k * delta)
    #         elif gradient_design == 2 and perturbation_type in [1, 2]:
    #             _of_m = self._sum_of_values(of_values=of_m)
    #             _of_p = self._sum_of_values(of_values=of_p)
    #             ghat_terms[:, 0] += (_of_p - _of_m) / (2.0 * c_k * delta)
    #         elif gradient_design == 1 and perturbation_type in [3]:
    #             _of_m = self._sum_of_values(of_values=of_m)
    #             _of = self._sum_of_values(of_values=of)
    #             ghat_terms[:, 0] += (of - of_m) / (c_k * delta[i])
    #         elif gradient_design == 2 and perturbation_type in [3]:
    #             _of_m = self._sum_of_values(of_values=of_m)
    #             _of_p = self._sum_of_values(of_values=of_p)
    #             ghat_terms[:, 0] += (of_p - of_m) / (2.0 * c_k * delta[i])
    #         else:
    #             # Otherwise raise error
    #             raise ValueError(
    #                 "Gradient design and perturbation type wrongly specified"
    #             )
    #     ghat = np.zeros(
    #         shape=(vec_size,),
    #         dtype=float,
    #     )
    #     # Calculate the mean of the gradient replications
    #     if gradient_design in [1, 2] and perturbation_type in [1, 2]:
    #         ghat_terms[:, 0] /= gradient_replications
    #     ghat += ghat_terms[:, 0]
    #     # Calculate the magnitude
    #     magnitude = np.linalg.norm(ghat)
    #     return ghat, magnitude


    # TODO: Duplicate in SPSA class. Merge
    def _gradient_odmat(self, odmat: pd.DataFrame) -> pd.Series:
        if self.seed_odmat is not None:
            df_odmat = self.seed_odmat.merge(
                odmat,
                on=["begin", "end", "from", "to"],
                how="outer",
            )
            # NOTE: 'value_x' will be the given, constant, input seed demands
            # NOTE: 'value_y' will be the estimated demands
            numerator = df_odmat["value_y"].sub(df_odmat["value_x"])
            denominator = np.linalg.norm(numerator, ord=2) * np.linalg.norm(
                df_odmat["value_x"], 
                ord=2,
            )
            odmat_vec_grad =  numerator / denominator
            return odmat_vec_grad
        else:
            raise ValueError("TODO")

    def approximate_gradient(
        self,
        x_vec: pd.DataFrame,
        extra_data: Dict[str, Any],
    ) -> Tuple[np.ndarray, float]:
        """Approximate the gradient through one-sided or two-sided finite difference \
            approximations.

        Args:
            x (numpy.ndarray): The current approximate solution.
            of (Dict[str, Any]): The corresponding objective function value(s).
            c_k (numpy.float64): The perturbation parameter at iteration k. This \
                parameter is used to scale the perturbation vector.

        Returns:
            Tuple[numpy.ndarray, numpy.float64]: The approximated gradient along with \
                its magnitude.
        """
        of = extra_data["of_values"]
        c_k = extra_data["c_k"]
        if self._estimator_options is not None:
            # Fetch some required and optional settings
            weight: Dict[str, float] = self._estimator_options.get_value(
                "weight_configuration"
            )

            gradient_design = self._estimator_options.get_value("gradient_design")
            perturbation_type = self._estimator_options.get_value(
                "perturbation_type"
            )
            gradient_replications = self._estimator_options.get_value(
                "gradient_replications"
            )
            vec_size = self._estimator_options.get_value("x0_size")
            # Instantiate a vector that is going to hold the gradient estimate
            ghat_terms = np.zeros(
                shape=(vec_size, 1),
                dtype=float,
            )
            # Estimate the gradient at the current approximate solution:
            for i in range(gradient_replications):
                # Perturbation vector type:
                #   1. Random perturbations
                if perturbation_type == 1:
                    delta = self.bernoulli_perturbation()
                #   2. Deterministic perturbations
                elif perturbation_type == 2:
                    delta = self.hadamard_perturbation()
                #   3. Coordinate perturbations (equivalent to using the FDSA algorithm)
                elif perturbation_type == 3:
                    # Generate the i'th unit vector
                    delta = self.coordinate_perturbation(int(i))
                else:
                    raise ValueError("TODO")
                # Approximate the gradient using a one-sided finite difference approximation
                if gradient_design == 1:
                    of_m = self.gradient_design_1(x_vec, c_k, delta)
                # Approximate the gradient using a two-sided finite difference approximation
                elif gradient_design == 2:
                    of_m, of_p = self.gradient_design_2(x_vec, c_k, delta)
                else:
                    raise ValueError("TODO")
                
                # TODO: Hardcoded for now, but would be better if more general
                if weight["odmat"] > 0:
                    ghat_terms[:, 0] += weight["odmat"] * self._gradient_odmat(
                        odmat=x_vec
                    )

                # Approximate the gradient
                if gradient_design == 1 and perturbation_type in [1, 2]:
                    # _of_m = self._sum_of_values(of_values=of_m)
                    # _of = self._sum_of_values(of_values=of)
                    # TODO: Hardcoded for now, but would be better if more general
                    _of_m = of_m["counts"][0] * of_m["counts"][1]
                    _of = of["counts"][0] * of["counts"][1]
                    ghat_terms[:, 0] += (_of - _of_m) / (c_k * delta)
                elif gradient_design == 2 and perturbation_type in [1, 2]:
                    # _of_m = self._sum_of_values(of_values=of_m)
                    # _of_p = self._sum_of_values(of_values=of_p)
                    # TODO: Hardcoded for now, but would be better if more general
                    _of_m = of_m["counts"][0] * of_m["counts"][1]
                    _of_p = of_p["counts"][0] * of_p["counts"][1]
                    ghat_terms[:, 0] += (_of_p - _of_m) / (2.0 * c_k * delta)
                elif gradient_design == 1 and perturbation_type in [3]:
                    # _of_m = self._sum_of_values(of_values=of_m)
                    # _of = self._sum_of_values(of_values=of)
                    # TODO: Hardcoded for now, but would be better if more general
                    _of_m = of_m["counts"][0] * of_m["counts"][1]
                    _of = of["counts"][0] * of["counts"][1]
                    ghat_terms[:, 0] += (_of - _of_m) / (c_k * delta[i])
                elif gradient_design == 2 and perturbation_type in [3]:
                    # _of_m = self._sum_of_values(of_values=of_m)
                    # _of_p = self._sum_of_values(of_values=of_p)
                    # TODO: Hardcoded for now, but would be better if more general
                    _of_m = of_m["counts"][0] * of_m["counts"][1]
                    _of_p = of_p["counts"][0] * of_p["counts"][1]
                    ghat_terms[:, 0] += (_of_p - _of_m) / (2.0 * c_k * delta[i])
                else:
                    # Otherwise raise error
                    raise ValueError(
                        "Gradient design and perturbation type wrongly specified"
                    )
            ghat = np.zeros(
                shape=(vec_size,),
                dtype=float,
            )
            # Calculate the mean of the gradient replications
            if gradient_design in [1, 2] and perturbation_type in [1, 2]:
                ghat_terms[:, 0] /= gradient_replications
            ghat += ghat_terms[:, 0]
            # Calculate the magnitude
            magnitude = np.linalg.norm(ghat)
            return ghat, magnitude
        else:
            raise ValueError("TODO")

    def gradient_design_1(
        self,
        x_vec: pd.DataFrame,
        c_k: float,
        delta: np.ndarray,
    ) -> Dict[str, Tuple[float, float]]:
        """Approximate the gradient using one-sided differences. One objective \
            function evaluation is used for this purpose.

        Args:
            x (numpy.ndarray): The current approximate solution.
            c_k (numpy.float64): The perturbation parameter at iteration k. This \
                parameter is used to scale the perturbation vector.
            delta (numpy.ndarray): The perturbation vector i.e., a vector of \
                elements that are used to perturb the elements of the current \
                approximate solution s.t. we can eventually approximate the gradient.

        Returns:
            Dict[str, numpy.float64]: The objective function value(s) obtained \
                from evaluating a single (positively) perturbed approximate solution \
                vector.
        """
        if self.models is not None and self.estimation_task is not None:
            # Minus perturbation
            x_m = self.create_perturbation(x_vec, c_k, delta, "minus")

            # Corresponding objective function value
            # start_sysusr = time.process_time()
            # start_wall = time.time()
            of_m, _, _ = self.objective_function(x_vec=x_m)
            # end_sysusr = time.process_time()
            # end_wall = time.time()

            # # Update the objective function evaluation counter
            # dict_ = {
            #     "estimation_task": self.estimation_task,
            # }
            # self.iteration_obj = self.models["Iteration"].objects.create(**dict_)
            # # Update the objective function evaluation counter
            # dict_ = {"eval_counter": self.estimation_task.eval_counter + 1}
            # self.models["EstimationTask"].objects.filter(
            #     uid=self.estimation_task.uid
            # ).update(**dict_)
            # self.estimation_task = self.models["EstimationTask"].objects.get(
            #     uid=str(self.estimation_task.uid)
            # )
            # dict_ = {
            #     "estimation_task": self.estimation_task,
            #     "iteration": self.iteration_obj,
            #     "of": of_m,
            #     # Save the (wall) time it took to run the objective function.
            #     "wall_time": end_wall - start_wall,
            #     # Save the (system + user) time it took to run the objective function.
            #     "sysusr_time": end_sysusr - start_sysusr,
            #     "simulation_data": simulator_object._to_json(),
            # }
            # self.of_obj = self.models["ObjectiveFunctionEvaluation"].objects.create(
            #     **dict_
            # )
            return of_m
        else:
            raise ValueError("TODO")

    def gradient_design_2(
        self,
        x_vec: pd.DataFrame,
        c_k: float,
        delta: np.ndarray,
        ) -> Tuple[
            Dict[str, Tuple[float,float]], Dict[str, Tuple[float,float]]
        ]:
        """Approximate the gradient using two-sided differences. Two objective \
            function evaluation are used for this purpose.


        Args:
            x (numpy.ndarray): The current approximate solution.
            c_k (numpy.float64): The perturbation parameter at iteration k. This \
                parameter is used to scale the perturbation vector.
            delta (numpy.ndarray): The perturbation vector i.e., a vector of elements \
                that are used to perturb the elements of the current approximate \
                solution s.t. we can eventually approximate the gradient.

        Returns:
            Tuple[Dict[str, numpy.float64], Dict[str, numpy.float64]]: The objective \
                function value(s) obtained from evaluating a positively and negatively \
                perturbed approximate solution vector.
        """
        if self.models is not None and self.estimation_task is not None:
            # Minus perturbation
            x_m = self.create_perturbation(x_vec, c_k, delta, "minus")
            # Corresponding objective function value
            # start_sysusr = time.process_time()
            # start_wall = time.time()
            of_m, _, _ = self.objective_function(x_vec=x_m)
            # end_sysusr = time.process_time()
            # end_wall = time.time()
            # dict_ = {"eval_counter": self.estimation_task.eval_counter + 1}
            # self.models["EstimationTask"].objects.filter(
            #     uid=self.estimation_task.uid
            # ).update(**dict_)
            # self.estimation_task = self.models["EstimationTask"].objects.get(
            #     uid=str(self.estimation_task.uid)
            # )
            # dict_ = {
            #     "estimation_task": self.estimation_task,
            #     "iteration": self.iteration_obj,
            #     "of": of_m,
            #     # Save the (wall) time it took to run the objective function.
            #     "wall_time": end_wall - start_wall,
            #     # Save the (system + user) time it took to run the objective 
            #     # function.
            #     "sysusr_time": end_sysusr - start_sysusr,
            #     "simulation_data": simulator_object._to_json(),
            # }
            # self.of_obj = self.models["ObjectiveFunctionEvaluation"].objects.create(
            #     **dict_
            # )

            # Plus perturbation
            x_p = self.create_perturbation(x_vec, c_k, delta, "plus")
            
            # Corresponding objective function value
            # start_sysusr = time.process_time()
            # start_wall = time.time()
            of_p, _, _ = self.objective_function(x_vec=x_p)
            # end_sysusr = time.process_time()
            # end_wall = time.time()

            # Update the objective function evaluation counter
            # dict_ = {"eval_counter": self.estimation_task.eval_counter + 1}
            # self.models["EstimationTask"].objects.filter(
            #     uid=self.estimation_task.uid
            # ).update(**dict_)
            # self.estimation_task = self.models["EstimationTask"].objects.get(
            #     uid=str(self.estimation_task.uid)
            # )
            # dict_ = {
            #     "estimation_task": self.estimation_task,
            #     "iteration": self.iteration_obj,
            #     "of": of_p,
            #     # Save the (wall) time it took to run the objective function.
            #     "wall_time": end_wall - start_wall,
            #     # Save the (system + user) time it took to run the objective function.
            #     "sysusr_time": end_sysusr - start_sysusr,
            #     "simulation_data": simulator_object._to_json(),
            # }
            # self.of_obj = self.models["ObjectiveFunctionEvaluation"].objects.create(
            #     **dict_
            # )
            return of_m, of_p
        else:
            raise ValueError("TODO")

    def hadamard_perturbation(self) -> np.ndarray:
        """Create a hadamard perturbation vector.

        Raises:
            ValueError: If gradient design option not available.

        Returns:
            numpy.ndarray: The perturbation vector i.e., a vector of elements \
                that are used to perturb the elements of the current approximate \
                solution s.t. we can approximate the gradient.
        """
        if self.estimation_task is not None and \
            self._estimator_options is not None: 
            # Fetch some required and optional settings
            gradient_design = self._estimator_options.get_value("gradient_design")
            vec_size = self._estimator_options.get_value("x0_size")
            matrix_order = self._estimator_options.get_value("matrix_order")
            hadamard_matrix = self._estimator_options.get_value("hadamard_matrix")
            # Fetch some current optimization results
            k = self.estimation_task.iter_counter
            # Generate a hardamard matrix: For one-sided finite difference approximations
            if gradient_design == 1:
                delta = hadamard_matrix[1 : vec_size + 1, k % matrix_order]
            # Generate a hardamard matrix: For two-sided finite difference approximations
            elif gradient_design == 2:
                delta = hadamard_matrix[0:vec_size, k % matrix_order]
            else:
                raise ValueError(
                    "Gradient design option not available. Only 'gradient_design' "
                    + "== 1 or == 2 is possible."
                )
            return delta
        else:
            raise ValueError("TODO")

    def bernoulli_perturbation(self) -> np.ndarray:
        """Generate a perturbation vector where the components take the value -1 or 1 \
            with probability 0.5. In other words, the comonents are sampled according \
            to a Rademacher distribution.

        Returns:
            delta (numpy.ndarray): The perturbation vector i.e., a vector of elements \
                that are used to perturb the elements of the current approximate \
                solution s.t. we can approximate the gradient.
        """
        if self.estimation_task is not None and \
            self._estimator_options is not None:
            # Fetch some required and optional settings
            vec_size = self._estimator_options.get_value("x0_size")
            delta = np.random.randint(0, 2, vec_size) * 2 - 1
            return delta
        else:
            raise ValueError("TODO")

    def coordinate_perturbation(self, i: int) -> np.ndarray:
        """Generate a perturbation vector that consists of components with the \
            value 0 and a single component with the value 1 at a given index i.

        Args:
            i (numpy.int64): The variable/coordinate to perturb such that we we \
                can calculate the rate of change with respect to this single variable.

        Returns:
            delta (numpy.ndarray): The perturbation vector i.e., a vector of \
                elements that are used to perturb the elements of the current \
                approximate solution s.t. we can approximate the gradient.
        """
        if self._estimator_options is not None:
            # Fetch some required and optional settings
            vec_size = self._estimator_options.get_value("x0_size")
            delta = np.zeros(shape=(vec_size,), dtype=float)
            delta[i] = 1.0
            return delta
        else:
            raise ValueError("TODO")

    def _generate_poly_data(
        self,
        x_vec: pd.DataFrame,
        of_values: Dict[str, Tuple[float, float]],
        ghat: pd.Series,
        max_step_length: float,
        extra_data: Dict[str, Any],
        ) -> Dict[str, Any]:
        n_points = 2
        increment = max_step_length / n_points
        x_vals = [x_vec]
        of_vals = [of_values]
        step_lengths = [0.0]
        simdata_vals = [extra_data["simdata"]]
        points = [i * increment for i in range(1, n_points + 1)]
        for s in points:
            x_k = x_vec.copy()
            x_k["value"] = x_vec["value"] - s * ghat
            of_values_k, simdata_k, _ = self.objective_function(x_vec=x_k)
            of_vals.append(of_values_k)
            x_vals.append(x_k)
            step_lengths.append(s)
            simdata_vals.append(simdata_k)
        return {
            "x_vals": x_vals,
            "of_vals": of_vals,
            "simdata_vals": simdata_vals,
            "step_lengths": step_lengths,
        }


class AssignmentMatrixBasedTravelDemandEstimator(
    GradientBasedTravelDemandEstimator
):

    def __init__(
        self,
        scenario_data: ScenarioData,
        file_io: FileIO,
        scenario_options: Dict[str, Any],
        simulator_class,
        simulator_options: Dict[str, Any],
        estimator_options: Dict[str, Any],
        collect_directory: str,
        lower_bounds=None,
        upper_bounds=None
        ) -> None:
        super().__init__(
            scenario_data=scenario_data,
            file_io=file_io,
            scenario_options=scenario_options,
            simulator_class=simulator_class,
            simulator_options=simulator_options,
            estimator_class=self,
            estimator_options=estimator_options,
            collect_directory=collect_directory,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        self.models = {
            EstimationTaskMetaData.__name__: EstimationTaskMetaData,
            EstimationTask.__name__: EstimationTask,
            Iteration.__name__: Iteration,
            ObjectiveFunctionEvaluation.__name__: ObjectiveFunctionEvaluation,
        }
        metadata = self.models["EstimationTaskMetaData"].objects.create(
            **{
                "file_io": file_io,
                "scenario_data": scenario_data,                
            },
        )
        self.estimation_task = self.models["EstimationTask"].objects.create(
            **{
                "metadata": metadata,
            },
        )
    
    @staticmethod
    def get_options_handler():
        return AssignmentMatrixEstimatorOptions

    def _polynomial_regression(
        self,
        x_vec: pd.DataFrame,
        of_values: Dict[str, Tuple[float, float]],
        ghat: pd.Series,
        extra_data: Dict[str, Any],
        ):
        x_val, of_val, extra_data = super()._polynomial_regression(
            x_vec, of_values, ghat, extra_data,
        )
        reevaluate = extra_data["reevaluate"]
        if reevaluate is True:
            # Only evaluate new point if proxy function is used
            # and it has not previously been evaluated
            of_val, simdata, assignmat = self.objective_function(x_val)
        else:
            simdata = extra_data["simdata"]
            assignmat = extra_data["assignmat"]
        # Overwrite 'extra_data' with new simulation data 
        extra_data = {
            "simdata": simdata,
            "assignmat": assignmat,
        }
        return x_val, of_val, extra_data

    def _generate_poly_data(
        self,
        x_vec: pd.DataFrame,
        of_values: Dict[str, Tuple[float, float]],
        ghat: pd.Series,
        max_step_length: float,
        extra_data: Dict[str, Any],
        ) -> Dict[str, Any]:
        n_points = 2
        increment = max_step_length / n_points
        x_vals = [x_vec]
        of_vals = [of_values]
        step_lengths = [0.0]
        simdata_vals = [extra_data["simdata"]]
        assignmat_vals = [extra_data["assignmat"]]
        points = [i * increment for i in range(1, n_points + 1)]
        for s in points:
            x_k = x_vec.copy()
            x_k["value"] = x_vec["value"] - s * ghat
            of_values_k, simdata_k, assignmat_k = self.objective_function_proxy(
                x_vec=x_k,
                assignmat=extra_data["assignmat"],
            )
            of_vals.append(of_values_k)
            x_vals.append(x_k)
            step_lengths.append(s)
            simdata_vals.append(simdata_k)
            assignmat_vals.append(assignmat_k)
        return {
            "x_vals": x_vals,
            "of_vals": of_vals,
            "simdata_vals": simdata_vals,
            "assignmat_vals": assignmat_vals,
            "step_lengths": step_lengths,
        }

    def objective_function_proxy(
        self,
        x_vec: pd.DataFrame,
        assignmat: pd.DataFrame
        ) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame, pd.DataFrame]:
        simdata = self.calculate_simdata(x_vec=x_vec, assignmat=assignmat)
        of_values = self.evaluate_of_terms(
            x_vec=x_vec,
            simdata=simdata,
            assignmat=assignmat,
        )
        return of_values, simdata, assignmat

    def evaluate_of_terms(
        self,
        x_vec: pd.DataFrame,
        simdata: pd.DataFrame,
        assignmat: pd.DataFrame
        ) -> Dict[str, Tuple[float, float]]:
        return_dict = {}
        val_odmat = self.evaluate_odmat(odmat=x_vec)
        return_dict.update(val_odmat)
        val_simdata = self.evaluate_simdata(simdata=simdata)
        if val_simdata is not None:
            return_dict.update(val_simdata)
        return return_dict 

    def calculate_counts(
        self,
        x_vec: pd.DataFrame,
        assignmat: pd.DataFrame,
        ) -> pd.DataFrame:

        # Locate links equipped with sensors. We should only determine
        # the discrepancy between observed and simulated quantities on
        # links equipped with sensors
        df_assignmat = assignmat.loc[
            assignmat["link"].isin(
                self.scenario_data.network.det_subset["link"],
            )
        ]

        # NOTE: 'value_x' will be the given, constant, input seed demands
        # NOTE: 'value_y' will be the estimated demands
        # Arrange the assignment proportions contained in the approximate 
        # assignment matrix 'assignmat' by 'link_begin', 'link_end' and 'link'.
        # Then we can then we can multiply the correct proportions with the
        # link count differences 
        df = df_assignmat.merge(
            x_vec,
            on=["from", "to", "begin"],
            how="left",
        )
        df["counts"] = df["value_y"] * df["ratio"]

        # Aggregate data by link and time interval
        df = df.groupby(
            by=["link_begin", "link_end", "link"]
        ).agg({"counts": "sum"})
        df = df.reset_index().rename(
            columns={"link_begin": "begin", "link_end": "end"},
        )     
        return df

    def calculate_simdata(
        self,
        x_vec: pd.DataFrame,
        assignmat: pd.DataFrame
        ) -> pd.DataFrame:
        # TODO: Duplicate code here. Merge some code with the
        #       'simdata_derivative()' method
        counts = self.calculate_counts(
            x_vec=x_vec,
            assignmat=assignmat
        )
        return counts

    # TODO: Duplicate in SPSA class. Merge
    def _gradient_odmat(self, odmat: pd.DataFrame) -> pd.Series:
        if self.seed_odmat is not None:
            df_odmat = self.seed_odmat.merge(
                odmat,
                on=["begin", "end", "from", "to"],
                how="outer",
            )
            # NOTE: 'value_x' will be the given, constant, input seed demands
            # NOTE: 'value_y' will be the estimated demands
            numerator = df_odmat["value_y"].sub(df_odmat["value_x"])
            denominator = np.linalg.norm(numerator, ord=2) * np.linalg.norm(
                df_odmat["value_x"], 
                ord=2,
            )
            odmat_vec_grad =  numerator / denominator
            return odmat_vec_grad
        else:
            raise ValueError("TODO")

    def _gradient_counts(
        self,
        odmat: pd.DataFrame,
        assignmat: pd.DataFrame,
        simdata: pd.DataFrame,
        ) -> pd.Series:
        # If the assignment matrix 'assignmat' is None, then initially no demand
        # was assigned to the network and the change in the demands with respect
        # to the link counts in the network will thus all be 0.
        if assignmat is None:
            simdata_vec_grad = odmat.copy()
            simdata_vec_grad["value"] = 0.0
            return simdata_vec_grad["value"]
        else:
            df_simdata = self.true_simdata.merge(
                simdata,
                on=["link", "begin", "end"],
                how="outer",
            )
            df_simdata = df_simdata.rename(
                columns={"begin": "link_begin", "end": "link_end"},
            )

            # Locate links equipped with sensors. We should only determine
            # the discrepancy between observed and simulated quantities on
            # links equipped with sensors
            df_simdata = df_simdata.loc[
                df_simdata["link"].isin(
                    self.scenario_data.network.det_subset["link"],
                )
            ]

            # If the simulation output data produced for a certain input 'x_vec' 
            # specifying trips between origins and destionations do not produce 
            # data on certain links, then set these values to 0 by default
            df_simdata = df_simdata.fillna(0)

            # NOTE: 'counts_x' will be the given, constant, input counts
            # NOTE: 'counts_y' will be the estimated counts obtained through
            #        simulation
            numerator = df_simdata["counts_y"].sub(df_simdata["counts_x"])
            df_simdata["counts_diff"] = numerator

            # Arrange the assignment proportions contained in the approximate 
            # assignment matrix 'assignmat' by 'link_begin', 'link_end' and 
            # 'link'. Then we can then we can multiply the correct proportions
            # with the link count differences 
            df = assignmat.merge(
                df_simdata,
                on=["link_begin", "link_end", "link"],
                how="left",
            ).fillna(0)
            df["_"] = df["counts_diff"] * df["ratio"]

            # Calculate the value of expression that enter into the denominator
            # of the expression of the gradient
            denominator = np.linalg.norm(
                x=df["counts_diff"], 
                ord=2
            ) * np.linalg.norm(x=df["counts_x"], ord=2) 

            # Group the terms, by origin and departure interval, that enter into 
            # the numerator of the expression of the gradient 
            df = df.groupby(by=["from", "to", "begin"]).agg({"_": "sum"})

            # Merge the aggregated result into the dataframe containing the 
            # origin destination trip demands. Merge on left to keep the correct
            # structure of the dataframe / ordering of the variable 
            simdata_vec_grad = (odmat.merge(
                df,
                on=["from", "to", "begin"],
                how="left",
            )["_"] / denominator).fillna(0)
            return simdata_vec_grad

    def _gradient_simdata(
        self,
        odmat,
        assignmat: pd.DataFrame,
        simdata: pd.DataFrame,
        ) -> pd.Series:
        counts_grad = self._gradient_counts(
            odmat=odmat,
            assignmat=assignmat,
            simdata=simdata,
        )
        return counts_grad
    
    def _initialize(
        self,
        x_vec: pd.DataFrame,
    ) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Any]]:
        if self.models is not None and self.estimation_task is not None:
            # Create and save a new (approximate) solution object
            dict_ = {"estimation_task": self.estimation_task}
            self.iteration_obj = self.models["Iteration"].objects.create(**dict_)
            # Update the objective function evaluation counter
            dict_ = {"eval_counter": self.estimation_task.eval_counter}
            self.models["EstimationTask"].objects.filter(
                uid=self.estimation_task.uid,
            ).update(**dict_)
            self.estimation_task = self.models["EstimationTask"].objects.get(
                uid=str(self.estimation_task.uid),
            )       
            # Evaluate the starting point/initial guess
            of_values, simdata, assignmat = self.objective_function(x_vec=x_vec)
            return of_values, {
                "simdata": simdata,
                "assignmat": assignmat,
            }
        else:
            raise ValueError("TODO")

    def take_step(
        self,
        x_vec: pd.DataFrame,
        of_values: Dict[str, Tuple[float, float]],
        ghat: pd.Series,
        extra_data: Dict[str, Any],
        ) -> Tuple[
            pd.DataFrame, Dict[str, Tuple[float, float]], Dict[str, Any]
        ]:
        # Adjust the current approximate solution and evaluate it
        x_final, of_final, extra_data = self._polynomial_regression(
                x_vec=x_vec,
                of_values=of_values,
                ghat=ghat,
                extra_data=extra_data,
        )
        return x_final, of_final, extra_data

    def approximate_gradient(
        self,
        x_vec: pd.DataFrame,
        extra_data: Dict[str, Any]
        ) -> Tuple[pd.Series, float]:
        if self._estimator_options is not None:
            ghat_magnitude_atol = self._estimator_options.get_value(
                "ghat_magnitude_atol"
            )
            ghat = np.zeros(self.true_odmat.shape[0])
            #
            odmat_grad = self._gradient_odmat(odmat=x_vec)
            # odmat_mag = np.linalg.norm(x=odmat_grad, ord=2)
            # if not np.isclose(odmat_mag, 0, atol=ghat_magnitude_atol, rtol=0):
            #         odmat_grad /= odmat_mag
            #
            # print()
            # print("odmat_grad: ", odmat_grad)
            simdata_grad = self._gradient_simdata(
                odmat=x_vec,
                assignmat=extra_data["assignmat"],
                simdata=extra_data["simdata"],
            )
            # print("simdata_grad: ", simdata_grad)
            # print()
            # simdata_mag = np.linalg.norm(x=simdata_grad, ord=2)
            # if not np.isclose(simdata_mag, 0, atol=ghat_magnitude_atol, rtol=0):
            #     simdata_grad /= simdata_mag
            ghat += odmat_grad + simdata_grad
            # print("GHAT: \n", ghat)
            return ghat, np.linalg.norm(ghat, ord=2)
        else:
            raise ValueError("TODO")

    def collect_final_output(
        self,
        x_vec: pd.DataFrame,
        extra_data: Dict[str, Any],
        ) -> None:
        if (self.models is not None) and \
            (self.estimation_task is not None) and \
            (self._estimator_options is not None):
            # Fecth latest estimation task db object
            estimation_task = self.models["EstimationTask"].objects.get(
                uid=str(self.estimation_task.uid)
            )
            weight: Dict[str, float] = self._estimator_options.get_value(
                "weight_configuration"
            )
            try:
                if weight["odmat"] <= 0.0:
                    seedmat = None
                else:
                    seedmat = self._scenario_options.get_value("seedmat")
            except KeyError:
                seedmat = None
            simdata = extra_data["simdata"]
            merged_simdata = self.true_simdata.merge(
                simdata,
                on=["link", "begin", "end"],
                how="outer",
            )
            merged_odmat = self.true_odmat.merge(
                x_vec,
                on=["begin", "end", "from", "to"],
                how="outer",
            )
            output_filename = f"gradient-assignmat_{seedmat}"
            output = {
                "experiment_metadata": {
                    "method": {
                        "estimator": "assignmat",
                        "estimator_options": self._estimator_options.get_options(),
                    },
                    "seedmatrix": seedmat,
                },
                # Estimated and observed demands
                "merged_odmat": merged_odmat.to_json(),
                # Estimated and oberved quantities
                "merged_simdata": merged_simdata.to_json(),
                # The history of objective function values
                "of_history": [
                    of_eval.of for of_eval in estimation_task.of_evals.all()
                ],
                "timing": {
                    "wall_time": [extra_data["wall_time"]],
                    # Save the (system + user) time 
                    "sysusr_time": [extra_data["sysusr_time"]],
                },
            }
            output_filepath = os.path.join(
                self.collect_directory,
                f"{output_filename}.json",
            )
            with open(output_filepath, mode="w") as file_object:
                json.dump(output, file_object, indent=4, sort_keys=True)
        else:
            raise ValueError("TODO")

    def run(
        self,
        x_vec: pd.DataFrame,
        collect_final_output: bool = False,
        ) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
        start_sysusr = time.process_time()
        start_wall = time.time()
        if self._estimator_options is not None:
            # Set the given random seed for reproducibility purposes
            np.random.seed(
                self._estimator_options.get_value("random_seed")
            )
            ghat_magnitude_atol = self._estimator_options.get_value(
                "ghat_magnitude_atol"
            )
            if self.models is not None and self.estimation_task is not None:
                of_values, extra_data = self._initialize(x_vec=x_vec)
                # Enter the main algorithm loop
                while self.termination_criteria() is False:
                    # Update the objective function evaluation counter
                    dict_ = {"iter_counter": self.estimation_task.iter_counter + 1}
                    self.models["EstimationTask"].objects.filter(
                        uid=self.estimation_task.uid
                    ).update(**dict_)
                    self.estimation_task = self.models["EstimationTask"].objects.get(
                        uid=str(self.estimation_task.uid)
                    )  
                    # Create and save a new iteration object
                    dict_ = {
                        "estimation_task": self.estimation_task,
                    }
                    self.iteration_obj = self.models["Iteration"].objects.create(**dict_)
                    # Apply possible constraints and update the current approximate solution
                    x_vec = self.apply_constraints(x_vec=x_vec)
                    # Approximate the gradient
                    ghat, magnitude = self.approximate_gradient(
                        x_vec=x_vec,
                        extra_data=extra_data,
                    )
                    # Check if we have arrived at a local minimum
                    if not np.isclose(magnitude, 0, atol=ghat_magnitude_atol, rtol=0):
                        # Take a step in the descent direction
                        x_vec, of_values, extra_data = self.take_step(
                            x_vec=x_vec,
                            of_values=of_values,
                            ghat=ghat,
                            extra_data=extra_data,
                        )

                        print("Last of values       : ", of_values)
                        print("Sum of last of values: ", self._sum_of_values(of_values=of_values))
                        print()
                        print("X VEC: ")
                        print(x_vec)
                    else:
                        print("Last of values       : ", of_values)
                        print("Sum of last of values: ", self._sum_of_values(of_values=of_values))
                        print()
                        print("X VEC: ")
                        print(x_vec)
                        break
                end_sysusr = time.process_time()
                end_wall = time.time()
                if collect_final_output:
                    extra_data.update({
                        # Save the (wall) time
                        "wall_time": end_wall - start_wall,
                        # Save the (system + user) time 
                        "sysusr_time": end_sysusr - start_sysusr,
                    })
                    self.collect_final_output(
                        x_vec=x_vec,
                        extra_data=extra_data,
                    )
                return of_values, x_vec
            else:
                raise ValueError("TODO")
        else:
            raise ValueError("TODO")


class SampleGenerator(BaseDataOrganizer):

    def __init__(
        self,
        scenario_data: ScenarioData,
        file_io: FileIO,
        scenario_options: Dict[str, Any],
        simulator_class,
        simulator_options: Dict[str, Any],
        sample_directory: Union[None, str] = None,
        lower_bounds=None,
        upper_bounds=None,
        ) -> None:

        # TODO: Make sure to use existing sample directory if given as input
        super().__init__(
            scenario_data=scenario_data,
            file_io=file_io,
            scenario_options=scenario_options,
        )

        self.models = {
            EstimationTaskMetaData.__name__: EstimationTaskMetaData,
            EstimationTask.__name__: EstimationTask,
            Iteration.__name__: Iteration,
            ObjectiveFunctionEvaluation.__name__: ObjectiveFunctionEvaluation,
        }
        metadata = self.models["EstimationTaskMetaData"].objects.create(
            **{
                "file_io": file_io,
                "scenario_data": scenario_data,                
            },
        )
        self.estimation_task = self.models["EstimationTask"].objects.create(
            **{
                "metadata": metadata,
            },
        )

        self.simulator = simulator_class

        # Set simulator options
        simulator_options_handler = simulator_class.get_options_handler()
        self._simulator_options = simulator_options_handler(simulator_options)

        # Set lower and upper bounds
        self.lower_bounds = self._set_lower_bounds(lower_bounds=lower_bounds)
        self.upper_bounds = self._set_upper_bounds(upper_bounds=upper_bounds)

        self.bounds = np.array(
            [   # Set upper and lower bounds on trip demands
                [self.lower_bounds[i], self.upper_bounds[i]] \
                for i in range(0, self.shape)
            ]
        )

    def run_simulator(self, x_vec: pd.DataFrame):
        _dict = {
            "scenario_data": self.scenario_data,
            "file_io": self.file_io,
            "simulator_options": self._simulator_options.get_options(),
        }
        simulator_object = self.simulator(**_dict) # TODO: Save object to db?
        _, simdata, assignmat, _ = simulator_object.run_simulator(x_vec=x_vec)
        return simdata, assignmat, simulator_object
    
    def _sample_metadata_exists(self) -> None:
        return None

    def generate_samples(
        self,
        n_samples: int,
        start_counter: int = 0
        ) -> None:
        sysusr_times = []
        wall_times = []

        # Check if sample metadata already exists and determine if a new 
        # batch of samples should be generated or new samples should be added
        # to an existing batch of already generated samples
        self._sample_metadata_exists()

        x_vec = ut.generate_constant_demand_vector(
            scenario_data=self.scenario_data,
            value=0.0,
        )
        x_vec = ut.construct_x_vec(
            scenario_data=self.scenario_data,
            demand_vec=x_vec,
        )
        self.n_samples = n_samples + start_counter

        if start_counter > 0:
            self.file_io.directory_counter = start_counter
            self.file_io.save()
        while start_counter <= self.n_samples:
            start_sysusr = time.process_time()
            start_wall = time.time()

            # Write out the current progress
            # TODO: Use 'logging.info' to easier be able to set msg verbosity
            print(
                f"\nSampling counter: {start_counter}. Remaining samples: " + \
                f"{self.n_samples - start_counter}\n"
            )
            v = sobol_sequence.sample(start_counter + 1, self.shape)
            scale_samples(v, problem={"bounds": self.bounds})
            # Unpack nested array
            x_vec["value"] = v[-1]
            self.run_simulator(x_vec=x_vec)
            start_counter += 1
            end_sysusr = time.process_time()
            end_wall = time.time()
            sysusr_times.append(end_sysusr - start_sysusr)
            wall_times.append(end_wall - start_wall)

        _dict = {
            "start_counter": start_counter,
            "n_samples": self.n_samples,
            # Save the (wall) time
            "wall_time": wall_times,
            # Save the (system + user) time 
            "sysusr_time": sysusr_times,
        }
        self.collect_sample_metadata(output=_dict)

    def collect_sample_metadata(self, output: Dict[str, Any]) -> None:
        # TODO: Is it even necessary to check this here? 
        if (self.models is not None):
            # TODO: Set filename globally, so it can be used consistently 
            # throughout the file
            output_filename = "sample_metadata"
            output_filepath = os.path.join(
                self.file_io.root_output_directory,
                f"{output_filename}.json",
            )
            if os.path.exists(output_filepath):
                raise ValueError("TODO: Update metadata")
            with open(output_filepath, mode="w") as file_object:
                json.dump(output, file_object, indent=4, sort_keys=True)
        else:
            raise ValueError("TODO")


class SurrogateModelBasedTravelDemandEstimator(BaseTravelDemandEstimator):

    def __init__(
        self,
        scenario_data: ScenarioData,
        file_io: FileIO,
        scenario_options: Dict[str, Any],
        simulator_class,
        simulator_options: Dict[str, Any],
        estimator_options: Dict[str, Any],
        collect_directory: str,
        lower_bounds=None,
        upper_bounds=None
        ) -> None:
        super().__init__(
            scenario_data=scenario_data,
            file_io=file_io,
            scenario_options=scenario_options,
            simulator_class=simulator_class,
            simulator_options=simulator_options,
            estimator_class=self,
            estimator_options=estimator_options,
            collect_directory=collect_directory,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
        )
        self.models = {
            EstimationTaskMetaData.__name__: EstimationTaskMetaData,
            EstimationTask.__name__: EstimationTask,
            Iteration.__name__: Iteration,
            ObjectiveFunctionEvaluation.__name__: ObjectiveFunctionEvaluation,
        }
        metadata = self.models["EstimationTaskMetaData"].objects.create(
            **{
                "file_io": file_io,
                "scenario_data": scenario_data,                
            },
        )
        self.estimation_task = self.models["EstimationTask"].objects.create(
            **{
                "metadata": metadata,
            },
        )

        self.bounds = np.array(
            [   # Set upper and lower bounds on trip demands
                [self.lower_bounds[i], self.upper_bounds[i]] \
                for i in range(0, self.shape)
            ]
        )

        # Create class variable for storing the surrogate model
        self.surrogate_model: Union[None, Callable] = None

        # Create empty list eventually used for storing the objective function
        # value history when optimizing on the surrogate model
        self.fval_history: List[float] = []
        # Internal use only to avoid repeated copying
        self.__copy_true_odmat = self.true_odmat.copy()
        self.__copy_true_simdata = self.true_simdata.copy()


    @staticmethod
    def get_options_handler():
        return SurrogateModelEstimatorOptions
    
    def set_surrogate_model(self, filename:str):
        return self.load_model(filename=filename)

    def collect_final_output_training(self):
        pass


    # def collect_final_output_optimization(
    #     self,
    #     x_vec: pd.DataFrame,
    #     extra_data: Dict[str, Any],
    #     ) -> None:
    #     if (self.models is not None) and \
    #         (self.estimation_task is not None) and \
    #         (self._estimator_options is not None):

    #         # Extract the sample data metadata and add it to the final 
    #         # output data obtained from the surrogate model experiment

    #         # TODO: Set filename globally, so it can be used consistently 
    #         # throughout the file
    #         sample_metadata_filename = "sample_metadata"
    #         output_filepath = os.path.join(
    #             self.file_io.root_output_directory,
    #             f"{sample_metadata_filename}.json",
    #         )
    #         # Set defaults
    #         sample_metadata = {"wall_time": 0.0, "sysusr_time": 0.0}
    #         if os.path.exists(output_filepath):
    #             with open(output_filepath, mode="r") as file_object:
    #                 sample_metadata = json.load(file_object)

    #         else:
    #             print(
    #                 f"Sample metadata '.json' file {sample_metadata_filename} \
    #                 does not exist!"
    #             )

    #         # Fecth latest estimation task db object
    #         estimation_task = self.models["EstimationTask"].objects.get(
    #             uid=str(self.estimation_task.uid)
    #         )
    #         weight: Dict[str, float] = self._estimator_options.get_value(
    #             "weight_configuration"
    #         )

    #         try:
    #             if weight["odmat"] <= 0.0:
    #                 seedmat = None
    #             else:
    #                 seedmat = self._scenario_options.get_value("seedmat")
    #         except KeyError:
    #             seedmat = None

    #         simdata = extra_data["simdata"]
    #         merged_simdata = self.true_simdata.merge(
    #             simdata,
    #             on=["link", "begin", "end"],
    #             how="outer",
    #         )
    #         merged_odmat = self.true_odmat.merge(
    #             x_vec,
    #             on=["begin", "end", "from", "to"],
    #             how="outer",
    #         )

    #         evaluate_output = self._estimator_options.get_value(
    #             "evaluate_output"
    #         )
    #         method = self._estimator_options.get_value("method")
    #         output_filename = f"surrogate-{method}-{evaluate_output}_{seedmat}"
    #         output = {
    #             "experiment_metadata": {
    #                 "method": {
    #                     "estimator": "assignmat",
    #                     "estimator_options": self._estimator_options.get_options(),
    #                 },
    #                 "seedmatrix": seedmat,
    #             },
    #             # Estimated and observed demands
    #             "merged_odmat": merged_odmat.to_json(),
    #             # Estimated and oberved quantities
    #             "merged_simdata": merged_simdata.to_json(),
    #             # The history of objective function values
    #             "of_history": [
    #                 of_eval.of for of_eval in estimation_task.of_evals.all()
    #             ],
    #             # The history of objective function values when optimizing on the
    #             # surrogate model
    #             "surrogate_of_history": self.fval_history,
    #             # Add the computational time used in each step of the applied
    #             # estimation method
    #             "timing": {
    #                 "run": {
    #                     "wall_time": [*extra_data["run"]["wall_time"]],
    #                     "sysusr_time": [*extra_data["run"]["sysusr_time"]],
    #                 },
    #                 "training": {
    #                     "wall_time": [*extra_data["training"]["wall_time"]],
    #                     "sysusr_time": [*extra_data["training"]["sysusr_time"]],
    #                 } ,
    #                 "optimization": {
    #                     "wall_time": [*extra_data["optimization"]["wall_time"]],
    #                     "sysusr_time": [*extra_data["optimization"]["sysusr_time"]],
    #                 }
    #             },
    #         }
    #         output_filepath = os.path.join(
    #             self.collect_directory,
    #             f"{output_filename}.json",
    #         )
    #         with open(output_filepath, mode="w") as file_object:
    #             json.dump(output, file_object, indent=4, sort_keys=True)
    #     else:
    #         raise ValueError("TODO")

    # def run(
    #     self, 
    #     x_vec: pd.DataFrame,
    #     collect_final_output: bool = False,
    #     train_only: bool = True,
    #     ) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
    #     if self._estimator_options is not None:
    #         np.random.seed(
    #             self._estimator_options.get_value("random_seed")
    #         )
    #         evaluate_output = self._estimator_options.get_value(
    #             "evaluate_output"
    #         )
    #         method = self._estimator_options.get_value("method")
    #         data = self.extract_data()

    #         train_start_sysusr = None; train_end_sysusr = None
    #         train_start_wall = None; train_end_wall = None
    #         if method == "knn":
    #             print(
    #                 f"Training a new '{method}:{evaluate_output}' model..."
    #             )
    #             self.surrogate_model, _ = self.set_surrogate_model(
    #                 filename=f"{method}_{evaluate_output}",
    #             )
    #             # If no existing model was loaded, then train a new one
    #             if self.surrogate_model is None:
    #                 train_start_sysusr = time.process_time()
    #                 train_start_wall = time.time()
    #                 self.surrogate_model = self.train_knn(
    #                     data=data,
    #                     evaluate_output=evaluate_output,
    #                 )
    #                 train_end_sysusr = time.process_time()
    #                 train_end_wall = time.time()
    #             else:
    #                 print(
    #                     f"An existing '{method}:{evaluate_output}' model was \
    #                     loaded..."
    #                 )
    #         elif method == "fnn":
    #             self.surrogate_model, _ = self.set_surrogate_model(
    #                 filename=f"{method}_{evaluate_output}",
    #             )
    #             if self.surrogate_model is None:
    #                 print(
    #                     f"Training a new '{method}:{evaluate_output}' model..."
    #                 )
    #                 train_start_sysusr = time.process_time()
    #                 train_start_wall = time.time()
    #                 self.surrogate_model = self.train_fnn(
    #                     data=data,
    #                     evaluate_output=evaluate_output,
    #                 )
    #                 train_end_sysusr = time.process_time()
    #                 train_end_wall = time.time()
    #             else:
    #                 print(
    #                     f"An existing '{method}:{evaluate_output}' model was \
    #                     loaded..."
    #                 )
    #         else:
    #             raise ValueError("TODO")
    #         optimize_start_sysusr = time.process_time() 
    #         optimize_start_wall = time.time() 
            
    #         if evaluate_output == "simdata_output":
    #             # Evaluate surrogate model modelling simulation output
    #             func = self._evaluate_model_predict_counts
    #         elif evaluate_output == "of_output":
    #             # Evaluate surrogate model modelling objective function output
    #             func = self._evaluate_model_predict_of
    #         else:
    #             raise ValueError("TODO")
    #         optimize_end_sysusr = time.process_time()
    #         optimize_end_wall = time.time()

    #         optimize_time_sysusr = optimize_end_sysusr - optimize_start_sysusr
    #         optimize_time_wall = optimize_end_wall - optimize_start_wall
            
    #         # Find minimum of surrogate model and evaluate it with sumo 
    #         of_values, x_vec, extra_data = self.optimize(func=func)

    #         if collect_final_output:
    #             if train_start_sysusr is None:
    #                 train_time_sysusr = train_end_sysusr - train_start_sysusr 
    #             if train_start_wall is None:
    #                 train_time_wall = train_end_wall - train_start_wall
 
    #             extra_data.update({
    #                 "run": {
    #                     "wall_time": 0.0,
    #                     "sysusr_time": 0.0,
    #                 },
    #                 "training": {
    #                     "wall_time": train_time_wall,
    #                     "sysusr_time": train_time_sysusr,
    #                 } ,
    #                 "optimization": {
    #                     "wall_time": optimize_time_wall,
    #                     "sysusr_time": optimize_time_sysusr,
    #                 },
    #                 # ...
    #                 "method": method,
    #                 "evaluate_output": evaluate_output,
    #             })
    #             self.collect_final_output(
    #                 x_vec=x_vec,
    #                 extra_data=extra_data,
    #             )
    #         return of_values, x_vec
    #     else:
    #         raise ValueError("TODO")




    def _training_step(
        self,
        collect_final_output: bool = False,
    ) -> None:
        if self._estimator_options is not None:
            np.random.seed(
                self._estimator_options.get_value("random_seed")
            )
            evaluate_output = self._estimator_options.get_value(
                "evaluate_output"
            )
            method = self._estimator_options.get_value("method")
            data = self.extract_data()

            train_start_sysusr = None; train_end_sysusr = None
            train_start_wall = None; train_end_wall = None
            if method == "knn":
                print(
                    f"Training a new '{method}:{evaluate_output}' model..."
                )
                train_start_sysusr = time.process_time()
                train_start_wall = time.time()
                self.surrogate_model = self.train_knn(
                    data=data,
                    evaluate_output=evaluate_output,
                )
                train_end_sysusr = time.process_time()
                train_end_wall = time.time()
            elif method == "fnn":
                print(
                    f"Training a new '{method}:{evaluate_output}' model..."
                )
                train_start_sysusr = time.process_time()
                train_start_wall = time.time()
                self.surrogate_model = self.train_fnn(
                    data=data,
                    evaluate_output=evaluate_output,
                )
                train_end_sysusr = time.process_time()
                train_end_wall = time.time()
            else:
                raise ValueError("TODO")

            if collect_final_output:
                train_time_sysusr = train_end_sysusr - train_start_sysusr 
                train_time_wall = train_end_wall - train_start_wall
 
                extra_data.update({
                    "run": {
                        "wall_time": 0.0,
                        "sysusr_time": 0.0,
                    },
                    "training": {
                        "wall_time": train_time_wall,
                        "sysusr_time": train_time_sysusr,
                    } ,
                    "optimization": {
                        "wall_time": 0.0,
                        "sysusr_time": 0.0,
                    },
                    # ...
                    "method": method,
                    "evaluate_output": evaluate_output,
                })
                self.collect_final_output_training_step(
                    # x_vec=x_vec,
                    extra_data=extra_data,
                )
        else:
            raise ValueError("TODO")


    def _optimization_step(
        self,
        collect_final_output: bool = False,
        ) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:

        np.random.seed(
            self._estimator_options.get_value("random_seed")
        )
        evaluate_output = self._estimator_options.get_value(
            "evaluate_output"
        )
        method = self._estimator_options.get_value("method")

        self.surrogate_model, _ = self.set_surrogate_model(
            filename=f"{method}_{evaluate_output}",
        )
        if self.surrogate_model is None:
            raise ValueError("TODO")

        if evaluate_output == "simdata_output":
            # Evaluate surrogate model modelling simulation output
            func = self._evaluate_model_predict_counts
        elif evaluate_output == "of_output":
            # Evaluate surrogate model modelling objective function output
            func = self._evaluate_model_predict_of
        else:
            raise ValueError("TODO")

        optimize_start_sysusr = time.process_time() 
        optimize_start_wall = time.time()             
        # Find minimum of surrogate model and evaluate it with sumo 
        of_values, x_vec, extra_data = self.optimize(func=func)
        optimize_end_sysusr = time.process_time()
        optimize_end_wall = time.time()

        optimize_time_sysusr = optimize_end_sysusr - optimize_start_sysusr
        optimize_time_wall = optimize_end_wall - optimize_start_wall

        if collect_final_output:

            extra_data.update({
                "run": {
                    "wall_time": 0.0,
                    "sysusr_time": 0.0,
                },
                "training": {
                    "wall_time": 0.0,
                    "sysusr_time": 0.0,
                } ,
                "optimization": {
                    "wall_time": optimize_time_wall,
                    "sysusr_time": optimize_time_sysusr,
                },
                # ...
                "method": method,
                "evaluate_output": evaluate_output,
            })
            self.collect_final_output_optimization_step(
                x_vec=x_vec,
                extra_data=extra_data,
            )
        return of_values, x_vec
    else:
        raise ValueError("TODO")






    def run(
        self, 
        x_vec: pd.DataFrame,
        collect_final_output: bool = False,
        train_only: bool = True,
        ) -> Tuple[Dict[str, Tuple[float, float]], pd.DataFrame]:
        if self._estimator_options is not None:
            np.random.seed(
                self._estimator_options.get_value("random_seed")
            )
            evaluate_output = self._estimator_options.get_value(
                "evaluate_output"
            )
            method = self._estimator_options.get_value("method")
            data = self.extract_data()

            train_start_sysusr = None; train_end_sysusr = None
            train_start_wall = None; train_end_wall = None
            if method == "knn":
                print(
                    f"Training a new '{method}:{evaluate_output}' model..."
                )
                self.surrogate_model, _ = self.set_surrogate_model(
                    filename=f"{method}_{evaluate_output}",
                )
                # If no existing model was loaded, then train a new one
                if self.surrogate_model is None:
                    train_start_sysusr = time.process_time()
                    train_start_wall = time.time()
                    self.surrogate_model = self.train_knn(
                        data=data,
                        evaluate_output=evaluate_output,
                    )
                    train_end_sysusr = time.process_time()
                    train_end_wall = time.time()
                else:
                    print(
                        f"An existing '{method}:{evaluate_output}' model was \
                        loaded..."
                    )
            elif method == "fnn":
                self.surrogate_model, _ = self.set_surrogate_model(
                    filename=f"{method}_{evaluate_output}",
                )
                if self.surrogate_model is None:
                    print(
                        f"Training a new '{method}:{evaluate_output}' model..."
                    )
                    train_start_sysusr = time.process_time()
                    train_start_wall = time.time()
                    self.surrogate_model = self.train_fnn(
                        data=data,
                        evaluate_output=evaluate_output,
                    )
                    train_end_sysusr = time.process_time()
                    train_end_wall = time.time()
                else:
                    print(
                        f"An existing '{method}:{evaluate_output}' model was \
                        loaded..."
                    )
            else:
                raise ValueError("TODO")
            
            if evaluate_output == "simdata_output":
                # Evaluate surrogate model modelling simulation output
                func = self._evaluate_model_predict_counts
            elif evaluate_output == "of_output":
                # Evaluate surrogate model modelling objective function output
                func = self._evaluate_model_predict_of
            else:
                raise ValueError("TODO")

            optimize_start_sysusr = time.process_time() 
            optimize_start_wall = time.time()             
            # Find minimum of surrogate model and evaluate it with sumo 
            of_values, x_vec, extra_data = self.optimize(func=func)
            optimize_end_sysusr = time.process_time()
            optimize_end_wall = time.time()

            optimize_time_sysusr = optimize_end_sysusr - optimize_start_sysusr
            optimize_time_wall = optimize_end_wall - optimize_start_wall

            if collect_final_output:
                if train_start_sysusr is None:
                    train_time_sysusr = train_end_sysusr - train_start_sysusr 
                if train_start_wall is None:
                    train_time_wall = train_end_wall - train_start_wall
 
                extra_data.update({
                    "run": {
                        "wall_time": 0.0,
                        "sysusr_time": 0.0,
                    },
                    "training": {
                        "wall_time": train_time_wall,
                        "sysusr_time": train_time_sysusr,
                    } ,
                    "optimization": {
                        "wall_time": optimize_time_wall,
                        "sysusr_time": optimize_time_sysusr,
                    },
                    # ...
                    "method": method,
                    "evaluate_output": evaluate_output,
                })
                self.collect_final_output(
                    x_vec=x_vec,
                    extra_data=extra_data,
                )
            return of_values, x_vec
        else:
            raise ValueError("TODO")

    def read_csv_files(self, dir) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load trip demands
        _types = {
            "value": float,
            "from": str,
            "to": str,
            "begin": float,
            "end": float,
            "slice": int,
        }
        odmat = pd.read_csv(
            os.path.join(dir, f"{ODMAT_FILENAME}.csv"),
            index_col=0,
            dtype=object,
        ).astype(_types)
        # Load simdata
        _types = {
            "link": str,
            "begin": float,
            "end": float,
            "counts": float,
        }
        simdata = pd.read_csv(
            os.path.join(dir, f"{SIMDATA_FILENAME}.csv"),
            index_col=0,
            dtype=object,
        ).astype(_types)
        return odmat, simdata

    def extract_data(
        self,
        processes: int = 6,
        ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        if self._estimator_options is not None:
            sample_directory = self._estimator_options.get_value(
                "sample_directory"
            )
            n_samples = self._estimator_options.get_value("n_samples")
            sample_dirs = [
                directory for directory in os.listdir(sample_directory) \
                if os.path.isdir(
                    os.path.join(sample_directory, directory)
                ) is True
            ]
            sample_dirs = sorted(sample_dirs, key=lambda x: int(x))

            # Limit the number of samples we read in based on given 'n_samples'
            # input value
            sample_dirs = sample_dirs[:n_samples]
            sample_dirs = [
                os.path.join(sample_directory, directory) 
                for directory in sample_dirs
            ]
            # Read in the files in parallel
            pool = Pool(processes=processes)
            data = pool.map(self.read_csv_files, [dir for dir in sample_dirs])
            return data
        else:
            raise ValueError("TODO")

    def train_fnn(
        self,
        data: List[Tuple[pd.DataFrame, pd.DataFrame]],
        evaluate_output: str,
        param_grid: Union[None, Dict[str, Any]] = None,
    ):
        if param_grid is None:
            # Use defaults if no param_grid as been passed
            hidden_layer_sizes = [
                # (
                #     int(1.00 * self.shape),
                #     int(1.00 * self.shape)
                # ),
                # (
                #     int(1.00 * self.shape),
                #     int(1.25 * self.shape)
                # ),
                # (
                #     int(1.25 * self.shape),
                #     int(1.00 * self.shape)
                # ),
                # (
                #     int(1.00 * self.shape),
                #     int(1.00 * self.shape), 
                #     int(1.00 * self.shape)
                # ),
                # (
                #     int(0.50 * self.shape),
                #     int(0.75 * self.shape),
                #     int(1.00 * self.shape),
                # ),
                (
                    int(0.75 * self.shape),
                    int(0.50 * self.shape),
                    int(0.50 * self.shape),
                ),
                (
                    int(0.75 * self.shape),
                    int(0.25 * self.shape),
                    int(0.25 * self.shape),
                ),
                # (
                #     int(0.25 * self.shape),
                #     int(0.25 * self.shape),
                #     int(0.75 * self.shape),
                # ),
                (
                    int(0.750 * self.shape),
                    int(0.125 * self.shape),
                    int(0.125 * self.shape),
                ),
                # (
                #     int(0.750 * self.shape),
                #     int(0.250 * self.shape),
                #     int(0.125 * self.shape),
                # ),
            ]
            alpha_range = [
                0.0001,
                0.001,
                0.01,
                0.1,
            ]
                # np.logspace(start=-4, stop=4, num=10, base=10)
            param_grid = {
                "reg__hidden_layer_sizes": hidden_layer_sizes,
                # "reg__activation": "relu",
                "reg__alpha": alpha_range,

            }
        pipe = Pipeline(
            [
                (
                    "reg",
                    MLPRegressor(
                        solver="adam",
                        max_iter=5000,
                        activation="relu", # TODO: Try tahn func.
                        learning_rate_init=0.01,
                        random_state=1
                    )
                )
            ]
        )
        return self.search(
            param_grid=param_grid,
            pipe=pipe,
            name="fnn",
            data=data,
            evaluate_output=evaluate_output,
        )

    def train_knn(
        self,
        data: List[Tuple[pd.DataFrame, pd.DataFrame]],
        evaluate_output: str,
        param_grid: Union[None, Dict[str, Any]] = None
        ):
        if param_grid is None:
            # Use defaults if no param_grid as been passed
            n_neighbors = list(set([
                int(value) for value in np.logspace(
                    start=2,
                    stop=6,
                    num=15,
                    base=2
                )
            ]))
            if np.max(n_neighbors) >= len(data):
                raise ValueError("More neighbours than number of data points")
            weights = ["uniform"] #, "distance"]
            # p = [1, 2]
            p = [2]
            param_grid = {
                "reg__n_neighbors": n_neighbors,
                "reg__weights": weights,
                "reg__p": p,
            }
        pipe = Pipeline([("reg", KNeighborsRegressor())])
        return self.search(
            param_grid=param_grid,
            pipe=pipe,
            name="knn",
            data=data,
            evaluate_output=evaluate_output,
        )

    # def gencon_penalty(self, x_vec: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    #     penalty = 0
    #     observed = []
    #     estimated = []
    #     for row_name in self.gencon:
    #         values = x_vec.loc[
    #             (x_vec["from"] == row_name),
    #             "value",
    #         ].to_frame(name="value")
    #         penalty += np.maximum(
    #             0,
    #             values["value"].sum() - self.gencon[row_name].iat[0],
    #         )
    #     return {
    #         "penalty": (0.0, penalty),
    #     }

    # def gencon_penalty(self, x_vec: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    #     penalty = 0; observed = []; estimated = []
    #     for row_name in self.gencon:
    #         values = x_vec.loc[
    #             (x_vec["from"] == row_name),
    #             "value",
    #         ].to_frame(name="value")
    #         observed.append(self.gencon[row_name].iat[0])
    #         max_estimated = np.maximum(
    #             self.gencon[row_name].iat[0],
    #             values["value"].sum(),
    #         )
    #         estimated.append(max_estimated)
    #     np_observed = np.array(observed)
    #     np_estimated = np.array(estimated)
    #     penalty = np.linalg.norm(
    #             x=(np_estimated - np_observed), 
    #             ord=2,
    #         ) / np.linalg.norm(
    #             x=np_observed,
    #             ord=2,
    #         )
    #     return {
    #         "penalty": (1.0, penalty),
    #     }

    # def gencon_penalty(self, x_vec: pd.DataFrame) -> float:
    def gencon_penalty(self, x) -> float:
        x_vec = self.true_odmat.copy()
        x_vec["value"] = x
        penalty = 0; observed = []; estimated = []
        for row_name in self.gencon:
            values = x_vec.loc[
                (x_vec["from"] == row_name),
                "value",
            ].to_frame(name="value")
            observed.append(self.gencon[row_name].iat[0])
            max_estimated = np.maximum(
                self.gencon[row_name].iat[0],
                values["value"].sum(),
            )
            estimated.append(max_estimated)
        np_observed = np.array(observed)
        np_estimated = np.array(estimated)
        penalty = np.linalg.norm(
                x=(np_estimated - np_observed), 
                ord=2,
            ) / np.linalg.norm(
                x=np_observed,
                ord=2,
            )
        return penalty

    def evaluate_of_terms(
        self,
        x_vec: pd.DataFrame,
        simdata: pd.DataFrame,
        assignmat: pd.DataFrame # TODO: Needed?
        ) -> Dict[str, Tuple[float, float]]:
        return_dict = {}
        val_odmat = self.evaluate_odmat(odmat=x_vec)
        return_dict.update(val_odmat)
        val_simdata = self.evaluate_simdata(simdata=simdata)
        if val_simdata is not None:
            return_dict.update(val_simdata)
        # val_penalty = self.gencon_penalty(x_vec=x_vec)
        # if val_penalty is not None:
        #     return_dict.update(val_penalty)
        # print("OF VALS: ", return_dict)
        return return_dict
    
    def _evaluate_model_predict_counts(self, x: np.ndarray) -> float:
        # x_vec = self.true_odmat.copy()
        # x_vec["value"] = x
        self.__copy_true_odmat["value"] = x
        x_reshaped = x.reshape(1, -1)
        if self.surrogate_model is not None:
            counts = self.surrogate_model.predict(x_reshaped)
            counts = counts.ravel()
            # simdata = self.true_simdata.copy()
            # simdata["counts"] = counts
            self.__copy_true_simdata["counts"] = counts
            of_values = self.evaluate_of_terms(
                # x_vec=x_vec,
                # simdata=simdata,
                x_vec=self.__copy_true_odmat,
                simdata=self.__copy_true_simdata,
                assignmat=None,
            )
            fval = self._sum_of_values(of_values=of_values)
            self.fval_history.append(fval)
            return fval
        else:
            raise ValueError("TODO")

    def _evaluate_model_predict_of(self, x: np.ndarray) -> float:
        if self._estimator_options is not None:
            weight: Dict[str, float] = self._estimator_options.get_value(
                "weight_configuration"
            )
            of_values = {}
            x_vec = self.true_odmat.copy()
            x_vec["value"] = x
            val_odmat = self.evaluate_odmat(odmat=x_vec)
            of_values.update(val_odmat)
            x_reshaped = x.reshape(1, -1)
            if self.surrogate_model is not None:
                of_value = self.surrogate_model.predict(x_reshaped)
                of_value = of_value.ravel()[0]
                val_simdata = {"simdata": (weight["counts"], of_value)}
                of_values.update(val_simdata)
                fval = self._sum_of_values(of_values=of_values)
                self.fval_history.append(fval)
                return fval
            else:
                raise ValueError("TODO")
        else:
            raise ValueError("TODO")

    def optimize(self, func: Callable) -> Tuple[
        Dict[str, Tuple[float, float]], pd.DataFrame, Dict[str, Any],
        ]:
        result = self._optimize(func=func)
        print("INFO: Optimization restults        : ")
        print("INFO: Objective function (OF) value: " + str(result.fun))
        x_vec = self.true_odmat.copy()
        x_vec["value"] = result.x
        x_vec = self.apply_constraints(x_vec=x_vec)
        print("result_x: \n", x_vec)
        
        # Evaluate the last solution
        of_values_new, simdata_new, assignmat_new = \
            self.objective_function(x_vec=x_vec)
        print("FINAL OF VALUES: ", of_values_new)
        print("            SUM: ", self._sum_of_values(of_values=of_values_new))
        extra_data = {
            "simdata": simdata_new,
            "assignmat": assignmat_new,
        }
        return of_values_new, x_vec, extra_data

    def _optimize(
        self,
        func: Callable,
        atol=0.001,
        tol=0.001,
        ) -> OptimizeResult:
        if self._estimator_options is not None:
            max_evals = self._estimator_options.get_value("max_evals")
            v = sobol_sequence.sample(max_evals, self.shape)
            scale_samples(v, problem={"bounds": self.bounds})
            # Unpack nested array
            new_sample = v[-1]
            minimizer_kwargs = {
                # "method": "COBYLA",
                "method": "SLSQP",
                "tol": 0.10,
                "constraints":  {
                    "type": "eq",
                    "fun": self.gencon_penalty
                },
            }
            # minimizer_kwargs = {
            #     "method": "L-BFGS-B",
            #     "bounds": self.bounds,
            #     "tol": 0.001,
            # }
            result = basinhopping(
                func=func,
                x0=new_sample,
                # niter=200,
                # niter=5,
                niter=2, # TODO: Temp.
                # T=0.10,
                # stepsize=1.0,
                minimizer_kwargs=minimizer_kwargs,
                seed=self._estimator_options.get_value("random_seed"),
            )
            return result
        else:
            raise ValueError("TODO")

        # if self._estimator_options is not None:
        #     result = differential_evolution(
        #         func=func,
        #         bounds=self.bounds,
        #         # workers=-1,
        #         # atol=atol,
        #         tol=tol,
        #         # updating="deferred",
        #         disp=True,
        #         maxiter=25,
        #         popsize=1,
        #         polish=False,
        #         init="sobol",
        #         seed=self._estimator_options.get_value("random_seed"),
        #     )
        #     return result
        # else:
        #     raise ValueError("TODO")

    def _organize_multi_target_model_data(self, data):
        _list = []
        loaded_odmat = self.true_odmat.copy()
        loaded_odmat["value"] = 0.0 

        loaded_simdata = self.true_simdata.copy()
        loaded_simdata["counts"] = 0.0
        
        of_values = self.evaluate_of_terms(
            x_vec=loaded_odmat,
            simdata=loaded_simdata,
            assignmat=None,
        )
        self._update_db_models(of_values=of_values)
        _list.append(self._sum_of_values(of_values=of_values))

        loaded_odmat = loaded_odmat.rename(columns={"value": "value0"})
        loaded_simdata = loaded_simdata.rename(columns={"counts": "counts0"})

        for i in range(len(data)):
            _df0 = data[i][0]
            _df1 = data[i][1]

            of_values = self.evaluate_of_terms(
                x_vec=_df0,
                simdata=_df1,
                assignmat=None,
            )
            self._update_db_models(of_values=of_values)
            _list.append(self._sum_of_values(of_values=of_values))

            _df0 = _df0.rename(columns={"value": f"value{i+1}"})
            loaded_odmat = loaded_odmat.merge(
                _df0,
                on=["begin", "end", "from", "to", "slice"],
                how="left",
            )

            _df1 = _df1.rename(columns={"counts": f"counts{i+1}"})
            loaded_simdata = loaded_simdata.merge(
                _df1,
                on=["begin", "end", "link"],
                how="left",
            )

        print("----> Best OF evaluation: ", np.min(_list))

        # Extract necessary columns in 'odmat' dataframe
        odmat_columns = loaded_odmat.columns[
            ~loaded_odmat.columns.isin(["from", "to", "begin", "end", "slice"])
        ]
        x = loaded_odmat[odmat_columns].fillna(0).values.transpose()

        # Extract necessary columns in 'simdata' dataframe
        simdata_columns = loaded_simdata.columns[
            ~loaded_simdata.columns.isin(["begin", "end", "link"])
        ]
        y = loaded_simdata[simdata_columns].fillna(0).values.transpose()
        return x, y

    def _organize_single_target_model_data(self, data):
        y = []
        loaded_odmat = self.true_odmat.copy()
        loaded_odmat["value"] = 0.0

        loaded_simdata = self.true_simdata.copy()
        loaded_simdata["counts"] = 0.0
        
        of_values = self.evaluate_of_terms(
            x_vec=loaded_odmat,
            simdata=loaded_simdata,
            assignmat=None,
        )
        self._update_db_models(of_values=of_values)          
        y.append(of_values["counts"][1])

        loaded_simdata = loaded_simdata.rename(columns={"counts": "counts0"})
        loaded_odmat = loaded_odmat.rename(columns={"value": "value0"})

        for i in range(len(data)):
            _df0 = data[i][0]
            _df1 = data[i][1]

            of_values = self.evaluate_of_terms(
                x_vec=_df0,
                simdata=_df1,
                assignmat=None,
            )
            self._update_db_models(of_values=of_values)
            y.append(of_values["counts"][1])

            _df0 = _df0.rename(columns={"value": f"value{i+1}"})
            loaded_odmat = loaded_odmat.merge(
                _df0,
                on=["begin", "end", "from", "to", "slice"],
                how="left",
            )

        # Extract necessary columns in 'odmat' dataframe
        # and transform to numpy arrays
        odmat_columns = loaded_odmat.columns[
            ~loaded_odmat.columns.isin(["from", "to", "begin", "end", "slice"])
        ]
        print("----> Best OF evaluation: ", np.min(y))
        x = loaded_odmat[odmat_columns].fillna(0).values.transpose()

        # Unravel
        y = np.array(y).reshape(-1, 1).ravel()
        return x, y

    def _update_db_models(
        self,
        of_values: Dict[str, Tuple[float, float]],
        ) -> None:
        if (self.models is not None) and \
            (self.estimation_task is not None):
            # Update the objective function evaluation counter
            dict_ = {
                "eval_counter": self.estimation_task.eval_counter + 1,
                "iter_counter": self.estimation_task.iter_counter + 1,
            }
            self.models["EstimationTask"].objects.filter(
                uid=self.estimation_task.uid
            ).update(**dict_)
            self.estimation_task = self.models["EstimationTask"].objects.get(
                uid=str(self.estimation_task.uid)
            )
            dict_ = {
                "estimation_task": self.estimation_task,
            }
            self.iteration_obj = self.models["Iteration"].objects.create(
                **dict_
            )
            dict_ = {
                "estimation_task": self.estimation_task,
                "iteration": self.iteration_obj,
                "of": of_values,
                # Save the (wall) time it took to evaluate the objective function.
                "wall_time": -1,
                # Save the (system + user) time it took to evaluate the objective 
                # function.
                "sysusr_time": -1,
                "simulation_data": json.dumps({}),
            }
            self.of_obj = self.models["ObjectiveFunctionEvaluation"].objects.create(
                **dict_
            )

    def search(
        self,
        param_grid: Dict[str, Any],
        pipe,
        name: str,
        data: List[Tuple[pd.DataFrame, pd.DataFrame]],
        n_iter: int = 25,
        cv: int = 5,
        n_jobs: int = -1,
        evaluate_output: str = "simdata_output",
        save_model: bool = True,
        ):
        grid = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=[
                "neg_mean_absolute_error",
                "neg_mean_squared_error",
            ],
            n_jobs=n_jobs,
            refit=False,
            # refit="neg_mean_squared_error",
            random_state=1,
            return_train_score=True,
            verbose=1,
        )
        if evaluate_output == "simdata_output":
            print(f"Fitting {name} multi-target model!")
            x, y = self._organize_multi_target_model_data(data=data)
        elif evaluate_output == "of_output":
            print(f"Fitting {name} single-target model!")
            x, y = self._organize_single_target_model_data(data=data)
        else:
            raise ValueError("TODO")
        fit = grid.fit(x, y)
        # Print and log some information about the fitted model
        print("Best Estimator Score     : ", grid.best_score_)
        print("Best Estimator Parameters: \n")
        for o in grid.best_params_:
            print("Parameter: " + str(o) + ", Value: " + str(grid.best_params_[o]))
        results = pd.DataFrame(grid.cv_results_)
        if save_model is True:
            self.save_model(
                grid=grid,
                metadata=results,
                filename=f"{name}_{evaluate_output}",
            )
        return grid
    
    def get_model_filepath(self, filename: str) -> str:
        if self._estimator_options is not None:
            sample_directory = self._estimator_options.get_value(
                "sample_directory"
            )
            model_filepath = os.path.join(sample_directory, filename)
            return model_filepath
        else:
            raise ValueError("TODO")

    def save_model(self, grid, metadata: pd.DataFrame, filename: str) -> None:
        model_filepath = self.get_model_filepath(filename=filename)
        pkl_filepath = f"{model_filepath}.pkl"
        csv_filepath = f"{model_filepath}_metadata.csv"
        if not os.path.exists(pkl_filepath):
            with open(pkl_filepath, "wb" ) as f:
                pickle.dump(grid, f)
        else:
            print(f"'.pkl' file: {pkl_filepath} already exists!")

        if not os.path.exists(csv_filepath):
            print("Overwriting existing '.csv' file!")
            # Save some model metadata as well
            metadata.to_csv(csv_filepath)
        else:
            print(f"'.csv' file: {csv_filepath} already exists!")

    def load_model(self, filename: str):
        filepath = self.get_model_filepath(filename="")
        pkl_file = os.path.join(filepath, f"{filename}.pkl")
        csv_file = os.path.join(filepath, f"{filename}_metadata.csv")
        if os.path.exists(pkl_file): #and os.path.exists(csv_file):
            with open(pkl_file, mode="rb") as f:
                model = pickle.load(f)
            # Load some model metadata as well
            metadata = pd.read_csv(csv_file)
            return model, metadata
        else:
            return None, None