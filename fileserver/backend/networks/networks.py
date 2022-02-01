from typing import Union, Dict, Any, List, Tuple
import logging
import os
import json
import uuid

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, Circle
from networkx.readwrite import json_graph

import backend.utils as ut


class BaseNetwork:

    def __init__(
        self,
        name: str,
        random_seed: int = 321,
        color_palette: str = "Blues_d",
        ) -> None:
        self.name: str = name
        # Set the random seed
        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.color_palette = color_palette
        self.uuid = str(uuid.uuid4()) 
        self._network: Union[
            nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph, None
        ] = None
        self._origins: Union[None, List] = None
        self._destinations: Union[None, List] = None
        self._od_pairs: Union[None, List] = None
    
    def get_network(self) -> Union[
            nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph, None
        ]:
        if self._network is not None:
            return self._network
        else:
            return None

    def report(self) -> None:

        if (self._origins is not None) and \
            (self._destinations is not None) and \
            (self._od_pairs is not None) and \
            (self._network is not None): 
            logging.info("Network Stats  ::")
            logging.info(f"Name            : {self.name}")
            logging.info(f"Origins         : {len(self._origins)}")
            logging.info(f"Destinations    : {len(self._destinations)}")
            logging.info(f"OD-Pairs        : {len(self._od_pairs)}")
            logging.info(f"Edges           : {len(self._network.edges)}")
            logging.info(f"Nodes           : {len(self._network.nodes)}")

            # These quantities should agree:
            edges_adjusted_origins = \
                len(self._network.edges) - len(self._origins) * 2
            logging.info(f"Edges (adjusted): {edges_adjusted_origins}")
            edges_adjusted_destinations = \
                len(self._network.edges) - len(self._destinations) * 2
            logging.info(f"Edges (adjusted): {edges_adjusted_destinations}")
            nodes_adjusted_origins = \
                len(self._network.nodes) - len(self._origins)
            logging.info(f"Nodes (adjusted): {nodes_adjusted_origins}")
            nodes_adjusted_destinations = \
                len(self._network.nodes) - len(self._destinations)
            logging.info(f"Nodes (adjusted): {nodes_adjusted_destinations}")
        else:
            logging.info("Network Stats  ::")
            logging.info("... No network has been genered yet ... ")
            
    def set_network_attributes(self) -> None:
        if self._network is not None:
            if isinstance(self._network, nx.MultiDiGraph):
                logging.info("Setting default network attributes")
                if not "crs" in self._network.graph:
                    default_crs = \
                        "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
                    self._network.graph["crs"] = default_crs
                if not "name" in self._network.graph:
                    self._network.graph["name"] = self.name
                if not "uuid" in self._network.graph:
                    self._network.graph["uuid"] = self.uuid
            else:
                raise ValueError("TODO")
        else:
            raise ValueError("TODO")

    def set_node_attributes(self) -> None:
        if self._network is not None:
            if isinstance(self._network, nx.MultiDiGraph):
                logging.info("Setting default node attributes")
                for node, data in self._network.nodes(data = True):
                    # Set whether the node should be remove or not
                    self._network.nodes[node]["remove_node"] = "False"
                    # Set the node in-degree
                    node_in_degree = self._network.in_degree(node)
                    self._network.nodes[node]["in_degree"] = node_in_degree
                    # Set the node out-degree
                    node_out_degree = self._network.out_degree(node)
                    self._network.nodes[node]["out_degree"] = node_out_degree
                    # Set the node degree
                    node_degree = node_in_degree + node_out_degree
                    self._network.nodes[node]["degree"] = node_degree
                    # Set node type for coloring (default type is 0)
                    self._network.nodes[node]["type"] = "0"
            else:
                raise ValueError("TODO")
        else:
            raise ValueError("TODO")

    def set_arc_attributes(self) -> None:
        if self._network is not None:
            if isinstance(self._network, nx.MultiDiGraph):
                logging.info("Setting default arc attributes")
                for u, v, data in self._network.edges(data = True):
                    # Set whether the arc should be remove or not
                    self._network.edges[(u, v, "0")]["remove_arc"] = "False"
                    # Set arc type for coloring (default type is 0)
                    self._network.edges[(u, v, "0")]["type"] = "0"
            else:
                raise ValueError("TODO")
        else:
            raise ValueError("TODO")

    def update_node_degrees(self) -> None:
        if self._network is not None:
            if isinstance(self._network, nx.MultiDiGraph):
                logging.info("Updating default node attributes")
                for node, data in self._network.nodes(data = True):
                    # Set whether the node should be remove or not
                    self._network.node[node]["remove_node"] = "False"
                    # Set the node in-degree
                    node_in_degree = self._network.in_degree(node)
                    self._network.nodes[node]["in_degree"] = node_in_degree
                    # Set the node out-degree
                    node_out_degree = self._network.out_degree(node)
                    self._network.nodes[node]["out_degree"] = node_out_degree
                    # Set the node degree
                    node_degree = node_in_degree + node_out_degree
                    self._network.nodes[node]["degree"] = node_degree
            else:
                raise ValueError("TODO")
        else:
            raise ValueError("TODO")

    def to_node_link_data(self) -> str:
        if self._network is not None:
            data_out = {}
            data_out["name"] = self.name
            data_out["uuid"] = self.uuid 
            data_out["_origins"] = self._origins
            data_out["_destinations"] = self._destinations
            data_out["_od_pairs"] = self._od_pairs
            data_out["network"] = json_graph.node_link_data(self._network)
            # return data_out
            return json.dumps(data_out)
        else:
            raise ValueError(
                "self._network is None. Method 'generate_network()' or \
                'from_node_link_data()' has not yet been called "
            )

    @staticmethod
    def from_node_link_data(data_in: str) -> Union[
            nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph,
        ]:
        data = json.loads(data_in)
        try:
            network = BaseNetwork(name="")
            network.name = data["name"]
            network.uuid = data["uuid"] 
            network._origins = data["_origins"]
            network._destinations = data["_destinations"]
            network._od_pairs = data["_od_pairs"]
            network._network = json_graph.node_link_graph(data["network"])
            return network 
        except KeyError as e:
            raise KeyError(e)

    def get_node_positions(self) -> Dict[Any, Any]:
        # Dict which hold the node positions
        pos = {}
        # scaling parameters
        xs = []; ys = []
        if self._network is not None:
            for node, data in self._network.nodes(data = True):
                xs.append(float(data["x"]))
                ys.append(float(data["y"]))
            max_x = np.max(xs); min_x = np.min(xs)
            max_y = np.max(ys); min_y = np.min(ys)
            # Extract and min/max scale node positions
            for node, data in self._network.nodes(data = True):
                pos[node] = np.array(
                    [
                        (float(data["x"]) - min_x) / (max_x - min_x) + 1000,
                        (float(data["y"]) - min_y) / (max_y - min_y)
                    ]
                )
            return pos
        else:
            logging.debug(
                "self._network is None. Method 'generate_network()' or \
                'from_node_link_data()' has not yet been called "                
            )
            raise ValueError("TODO")

    def get_edge_list(self) -> Union[None, List[Tuple[str, str]]]:
        if self._network is not None:
            return [(u, v) for u, v, _ in self._network.edges]
        else:
            logging.debug(
                "self._network is None. Method 'generate_network()' or \
                'from_node_link_data()' has not yet been called "                
            )
            return None

    def draw_network(
        self,
        network: Union[
            nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph, None
        ],
        nodes_pos_network: List,
        ax,
        ):
        # if self._network is not None:
        if network is not None:
            patches = {} 
            # for n in self._network:
            for n in network:
                if n in self._origins or n in self._destinations:
                    colors = sns.color_palette(self.color_palette)[2]
                    color = (colors[0], colors[1], colors[2], 1)
                else:
                    colors = sns.color_palette(self.color_palette)[0]
                    color = (colors[0], colors[1], colors[2], 1)
                c = Circle(
                    nodes_pos_network[n],
                    radius=0.015,
                    alpha=1.0,
                    color=color,
                )
                ax.add_patch(c)
                patches[n] = c
                x, y = nodes_pos_network[n]
            seen = {}
            # for (u, v, d) in self._network.edges(data=True):
            for (u, v, d) in network.edges(data=True):
                n1 = patches[u]
                n2 = patches[v]
                rad = 0.15
                if (u,v) in seen:
                    rad = seen.get((u,v))
                    rad = (rad + np.sign(rad) * 0.1) * -1
                # alpha = edges[(u,v)]
                if u in self._origins or u in self._destinations:
                    linestyle = "--"
                    colors = sns.color_palette(self.color_palette)[2]
                    color = (colors[0], colors[1], colors[2], 1)
                elif v in self._origins or v in self._destinations:
                    linestyle = "--"
                    colors = sns.color_palette(self.color_palette)[2]
                    color = (colors[0], colors[1], colors[2], 1)
                else:
                    linestyle = "-"
                    colors = sns.color_palette(self.color_palette)[0]
                    color = (colors[0], colors[1], colors[2], 1)
                e = FancyArrowPatch(
                    n1.center, n2.center,
                    patchA=n1, patchB=n2,
                    arrowstyle="-|>",
                    connectionstyle="arc3,rad=%s"%rad,
                    # mutation_scale=22.5,
                    mutation_scale=62.5, # Arrow size
                    # lw=1.75,
                    lw=3.00,
                    alpha=1.0,
                    color=color,
                    linestyle=linestyle,
                    capstyle="round",
                )
                seen[(u,v)] = rad
                ax.add_patch(e)
            return ax
        else:
            raise ValueError(
                "self._network is None. Method 'generate_network()' or \
                'from_node_link_data()' has not yet been called "
            )
    
    def visualize_network0(
        self,
        figname: str,
        figsize: Tuple[float, float] = (20, 20),
        wspace: float = 0.20,
        hspace: float = 0.20,
        ) -> Tuple[Any, Any]:
        if (self._network is not None) and \
            (self._origins is not None) and \
            (self._destinations is not None):
            network = self._network.copy()
            network.remove_nodes_from(
                self._origins + self._destinations,
            )
            nodes_pos_network = self.get_node_positions()
            fig, ax = plt.subplots(
                nrows=1, 
                ncols=1,
                sharex=True,
                sharey=True,
                num=1,
            )
            self.draw_network(
                network=network,
                nodes_pos_network=nodes_pos_network,
                ax=ax,
            )
            nx.draw_networkx_nodes(
                network,
                nodes_pos_network,
                nodelist=network.nodes,
                ax=ax,
                node_size=0,
                alpha=0.0,
            )
            ax.set_axis_off()
            sns.set_palette(sns.color_palette(self.color_palette))
            sns.set_style("darkgrid", {"axes.facecolor": "1.0"})
            fig.set_size_inches(figsize)
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
            sns.set_context("paper")
            plt.tight_layout()
            return fig, ax
        else:
            raise ValueError(
                "self._network is None. Method 'generate_network()' or \
                'from_node_link_data()' has not yet been called "
            )

    def visualize_network1(
        self,
        figname: str,
        figsize: Tuple[float, float] = (20, 20),
        wspace: float = 0.20,
        hspace: float = 0.20,
        ) -> Tuple[Any, Any]:
        if (self._network is not None) and \
            (self._origins is not None) and \
            (self._destinations is not None):
            nodes_pos_network = self.get_node_positions()
            edge_list_network = self.get_edge_list()
            fig, ax = plt.subplots(nrows = 1, ncols = 1, sharex = True, sharey = True, num = 1)
            self.draw_network(
                network=self._network,
                nodes_pos_network=nodes_pos_network,
                ax=ax,
            )
            nx.draw_networkx_nodes(
                self._network,
                nodes_pos_network,
                nodelist=self._network.nodes,
                ax=ax,
                node_size=0,
                alpha=0.00,
            )

            sns.set_palette(sns.color_palette("Blues_d")) 
            ax.set_axis_off()
            sns.set_palette(sns.color_palette(self.color_palette))
            sns.set_style("darkgrid", {"axes.facecolor": "1.0"})
            fig.set_size_inches(figsize)
            fig.subplots_adjust(wspace=wspace, hspace=hspace)
            sns.set_context("paper")
            plt.tight_layout()
            return fig, ax
        else:
            raise ValueError(
                "self._network is None. Method 'generate_network()' or \
                'from_node_link_data()' has not yet been called "
            )

    def visualize_network(
        self, 
        figname: str,
        plot_type: int,
        figsize: Tuple[float, float] = (20, 20),
        wspace: float = 0.20,
        hspace: float = 0.20,
        ) -> Tuple[Any, Any]:
        # without sources/sinks
        if plot_type == 0:
            return self.visualize_network0(
                figname=figname,
                figsize=figsize,
                wspace=wspace,
                hspace=hspace,
            )
        # with sources/sinks
        elif plot_type == 1:
            return self.visualize_network1(
                figname=figname,
                figsize=figsize,
                wspace=wspace,
                hspace=hspace,
            )
        else:
            raise ValueError("TODO")


class GridNetwork(BaseNetwork):

    def __init__(
        self,
        name: str,
        random_seed: int = 321,
        color_palette: str = "Blues_d",
        dim: int = 4,
        arc_length: float = 250,
        ) -> None:
        super().__init__(
            name=name,
            random_seed=random_seed,
            color_palette=color_palette,
        )
        if dim < 2:
            error = f"Invalid number of dimensions: {dim}. The dimension \
                should be larger than 1 i.e., dim > 1"
            logging.error(error)
            raise ValueError(error)
        self._dim: int = dim + 2
        self._arc_length: float = arc_length
        self._nodes: Union[None, List] = None
        self._node_positions: Union[None, List] = None
        self._links: Union[None, List] = None
        self._index_dict: Union[None, Dict] = None

    def is_corner_node(self, index_dict: Dict, index) -> bool:
        # Determine whether or not a certain node is a corner node
        # in the generated grid
        i = index_dict[index][0]
        j = index_dict[index][1]
        corner_node = (i == 0 and j == 0) or \
            (i == 1 and j == 0) or \
            (i == self._dim - 1 and j == 0) or \
            (i == self._dim - 2 and j == 0) or \
            (i == 0 and j == self._dim - 1) or \
            (i == 1 and j == self._dim - 1) or \
            (i == self._dim - 1 and j == self._dim - 1) or \
            (i == self._dim - 2 and j == self._dim - 1)
        return corner_node

    def generate_network(self) -> None:
        self._origins = []
        self._destinations = []
        self._links = []
        self._nodes = []
        self._node_positions = []
        self._index_dict = {}
        # Generate a grid network
        # Origins and destinations
        for i in range(0, self._dim):
            for j in range(0, self._dim):
                index = i * self._dim + j
                self._index_dict[index] = (i, j)
        for i in range(0, self._dim):
            for j in range(0, self._dim):
                index = i * self._dim + j
                # Exclude corner nodes
                if not self.is_corner_node(self._index_dict, index):
                    self._nodes.append(str(index))
                    self._node_positions.append(
                        (
                            (i + 1) * self._arc_length + \
                                np.random.randint(0, self._arc_length),
                            (j + 1) * self._arc_length + \
                                np.random.randint(0, self._arc_length)
                        )
                    )
                    if (i == 0 or i == self._dim - 1) or \
                        (j == 0 or j == self._dim - 1):
                        self._origins.append(str(index))
                        self._destinations.append(str(index))
                    # Connect rows
                    if i != 0 and i != self._dim - 1:
                        next_index = index + 1
                        is_corner_node = self.is_corner_node(
                            self._index_dict,
                            next_index
                        )
                        if not is_corner_node and \
                            (self._index_dict[next_index][0] == \
                                self._index_dict[index][0]):
                            self._links.append((str(index), str(index + 1)))
                            self._links.append((str(index + 1), str(index)))
                    # Connect columns
                    if i != self._dim - 1 and (j != 0 and j != self._dim - 1):
                        next_index = (i + 1) * self._dim + j
                        is_corner_node = self.is_corner_node(
                            self._index_dict,
                            next_index
                        )
                        if not is_corner_node:
                            self._links.append((str(index), str(next_index)))
                            self._links.append((str(next_index), str(index)))
        # Instantiate the network (a multi directed graph)
        self._network = nx.MultiDiGraph()
        # Add nodes and arcs to the network
        self.add_nodes(); self.add_arcs()
        # Set some default node and arc attributes
        self.set_node_attributes(); self.set_arc_attributes()
        # Set O-D pairs and default network attribues
        self.set_od_pairs(); self.set_network_attributes()
        # Report back some stats about the generated network
        self.report()

    def add_nodes(self) -> None:
        if (self._nodes is not None) and \
            (self._node_positions is not None) and \
            (self._network is not None):
            for i in range(0, len(self._nodes)):
                x = self._node_positions[i][0]
                y = self._node_positions[i][1]
                self._network.add_node(
                    self._nodes[i],
                    x = x,
                    y = y,
                    # geometry = Point(x, y),
                )
        else:
            raise ValueError("TODO")

    def add_arcs(
        self,
        priority: int = -1,
        maxspeed: float = 13.89,
        lanes: int = 1
        ) -> None:
        if (self._network is not None) and \
            (self._links is not None) and \
            (self._origins is not None) and \
            (self._destinations is not None):
            capacity = ut.get_capacity(priority, maxspeed, lanes)
            for arc in self._links:
                node1 = arc[0]; node2 = arc[1]
                # Create a straight line from one point to another
                x1 = float(self._network.nodes[node1]["x"])
                x2 = float(self._network.nodes[node2]["x"])
                y1 = float(self._network.nodes[node1]["y"])
                y2 = float(self._network.nodes[node2]["y"])
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Create 500m dummy arcs
                if node1 in self._origins + self._destinations:
                    x1 = float(self._network.nodes[node2]["x"]) + 353.556
                    y1 = float(self._network.nodes[node2]["y"]) + 353.556
                    self._network.nodes[node1]["x"] = x1
                    self._network.nodes[node1]["y"] = y1
                    # self._network.nodes[node1]["geometry"] = Point(x1, y1)
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # Create 500m dummy arcs
                elif node2 in self._origins + self._destinations:
                    x2 = float(self._network.nodes[node1]["x"]) + 353.556
                    y2 = float(self._network.nodes[node1]["y"]) + 353.556
                    self._network.nodes[node2]["x"] = x2
                    self._network.nodes[node2]["y"] = y2
                    # self._network.nodes[node2]["geometry"] = Point(x2, y2)
                    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                # geometry = LineString([Point(x1, y1), Point(x2, y2)])
                self._network.add_edge(
                    u_for_edge=node1,
                    v_for_edge=node2,
                    key="0",
                    # geometry=geometry,
                    length=length,
                    priority=priority,
                    # The 'maxspeed' attribute is also set in the network 
                    # converter class
                    maxspeed=maxspeed,
                    lanes=lanes,
                    capacity=capacity,
                )
        else:
            raise ValueError("TODO")

    def set_od_pairs(self) -> None:
        if (self._origins is not None) and \
            (self._destinations is not None):
            # Create an empty list, which is going to hold the O-D pairs
            self._od_pairs = []
            for origin in self._origins:
                for destination in self._destinations:
                    if origin != destination:
                        self._od_pairs.append((origin, destination))



class SumoNetworkOrganizer:

    def __init__(
        self,
        net_file: str,
        taz_file: Union[None, str] = None,
        det_subset_file: Union[None, str] = None
        ) -> None:
        self.net_file = net_file
        self.nodes: Union[None, pd.DataFrame] = None
        self.links: Union[None, pd.DataFrame] = None
        self.taz_file = taz_file
        self.taz: Union[None, pd.DataFrame] = None
        self.det_subset_file = det_subset_file
        self.det_subset: Union[None, pd.DataFrame] = None
        # SUMO specific variables only needed by SUMO
        self.sumo_det_all_file: Union[None, str] = None
        self.sumo_det_all: Union[None, pd.DataFrame] = None
        #
        self.links_not_in_taz: List = []
        self.links_in_taz: List = []
        
        self._nodes_link_map = {}

    def set_links_in_taz(self) -> None:
        if self.taz is not None:
            for _, row in self.taz.iterrows():
                for link in row["sources"]:
                    self.links_in_taz.append(link)
                for link in row["sinks"]:
                    self.links_in_taz.append(link)
        else:
            raise ValueError("TODO")

    def set_links_not_in_taz(self) -> None:
        if self.links is not None:
            for _, row in self.links.iterrows():
                if not row["edge"] in self.links_in_taz:
                    self.links_not_in_taz.append(row["edge"])
        else:
            raise ValueError("TODO")

    def initialize_network_object(self, simulator: str = "sumo") -> None:
        if simulator == "sumo":
            self.nodes, self.links = self.read_net_file_sumo()
        else:
            raise ValueError()
        if self.taz_file is not None:
            self.taz = self.read_taz_file(taz_file=self.taz_file)
            # Test - Start
            self.set_links_in_taz()
            self.set_links_not_in_taz()
            # Test - End
            self._nodes_link_map = self.get_nodes_link_map()
            if not (self.det_subset_file is None):
                self.det_subset = self.read_det_subset_file(
                    det_subset_file=self.det_subset_file,
                )
        else:
            raise ValueError("TODO")

    def read_net_file_sumo(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        root = ut.read_xml_file(self.net_file)
        nodes = self.get_nodes_sumo(root)
        links = self.get_links_sumo(root)
        return nodes, links

    @staticmethod
    def get_nodes_sumo(root) -> pd.DataFrame:
        nodes = []
        for child in root:
            if child.tag == "junction":
                junction_attributes = {}
                if child.attrib["type"] != "internal":
                    for attribute in child.attrib:
                        junction_attributes[attribute] = child.attrib[attribute]
                    nodes.append(junction_attributes)
        return pd.DataFrame(data=nodes, dtype=object)

    @staticmethod
    def get_links_sumo(root) -> pd.DataFrame:
        links = []
        for child in root:
            if ((child.tag == "edge" and "priority" in child.attrib) and \
            ("from" in child.attrib and "to" in child.attrib)):
                link_attributes = {}
                link_attributes["edge"] = child.attrib["id"]
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
                        link_attributes[item.attrib["key"]] = \
                            item.attrib["value"]
                link_attributes["lane"] = lane_id
                link_attributes["index"] = lane_index
                link_attributes["length"] = lane_length
                link_attributes["speed"] = lane_speed
                links.append(link_attributes)
        types = {"from": str, "to": str, "edge": str}
        df = pd.DataFrame(data=links, dtype=object).astype(types)
        return df

    @staticmethod
    def read_det_file(det_file: str) -> pd.DataFrame:
        root = ut.read_xml_file(det_file)
        det = []
        for child in root:
            if child.tag == "inductionLoop":
                det_attribs = {}
                for attribute in child.attrib:
                    det_attribs[attribute] = child.attrib[attribute]
                det.append(det_attribs)
        df = pd.DataFrame(data=det, dtype=object)
        return df

    @staticmethod
    def read_det_subset_file(det_subset_file: str) -> pd.DataFrame:
        root = ut.read_xml_file(det_subset_file)
        subset_det = []
        for child in root:
            if child.tag == "additional":
                for subchild in child:
                    if subchild.tag == "inductionLoop":
                        subset_det_attribs = {}
                        for attribute in subchild.attrib:
                            if attribute == "id":
                                subset_det_attribs["edge"] = \
                                    subchild.attrib["id"]
                            else:
                                subset_det_attribs[attribute] = \
                                    subchild.attrib[attribute]
                        subset_det.append(subset_det_attribs)
        df = pd.DataFrame(data=subset_det, dtype=object)
        return df

    # Meandata output for edges
    # def generate_det_file(self, sim_output_path, sim_output_dir, interval_increment):
    #     """ This method is only used by SUMO.
    #     """
    #     det_file = sim_output_dir + "/" + self.net_file.split("/")[-1].split(".")[0] + \
    #         ".det.xml"
    #     sim_output_path = sim_output_path.split("/")[-1]
    #     with open(det_file, "w") as file:
    #         file.write("<additional>\n")
    #         file.write("\t" + "<edgeData id=\"" + "0" + "\"" + \
    #             " freq=\"" + str(interval_increment) + "\"" + \
    #             " file=\"" + sim_output_path + "\"" + "/>\n")
    #         # for index, row in self.links.iterrows():
    #         #     for i in range(0, len(row["lane"])):
    #         #         file.write("\t" + "<edgeData id=\"" + row["lane"][i] + "\"" + \
    #         #         " freq=\"" + str(interval_increment) + "\"" + \
    #         #         " file=\"" + sim_output_path + "\"" + "/>\n")
    #         file.write("</additional>\n")
    #     self.sumo_det_all_file = det_file; self.sumo_det_all = self.read_det_file(det_file)

    # Using induction loops (detector placed in the middle of the segement)
    # def generate_det_file(self, sim_output_path, sim_output_dir, interval_increment):
    #     """ This method is only used by SUMO.
    #     """
    #     det_file = sim_output_dir + "/" + self.net_file.split("/")[-1].split(".")[0] + \
    #         ".det.xml"
    #     sim_output_path = sim_output_path.split("/")[-1]
    #     with open(det_file, "w") as file:
    #         file.write("<additional>\n")
    #         for index, row in self.links.iterrows():
    #             for i in range(0, len(row["lane"])):
    #                 file.write("\t" + "<inductionLoop id=\"" + row["lane"][i] + "\" lane=\"" + \
    #                     row["lane"][i] + "\" pos=\"" + str(np.float64(row["length"][i])/2.0) + \
    #                     "\" freq=\"" + str(interval_increment) + "\"" + \
    #                     " file=\"" + sim_output_path + "\"" + "/>\n")
    #         file.write("</additional>\n")
    #     self.sumo_det_all_file = det_file; self.sumo_det_all = self.read_det_file(det_file)

    # Using induction loops (detector placed in the end of the segement)
    def generate_det_file(
        self,
        sim_output_path: str,
        sim_output_dir: str,
        interval_increment: float,
        detector_type: str = "induction_loop",
        ) -> None:
        if detector_type == "induction_loop":
            self.generate_induction_loop_det_file(
                sim_output_path=sim_output_path,
                sim_output_dir=sim_output_dir,
                interval_increment=interval_increment,
            )
        else:
            raise ValueError("TODO")
    
    def generate_induction_loop_det_file(
        self,
        sim_output_path: str,
        sim_output_dir: str,
        interval_increment: float,
        ) -> None:
        if self.links is not None:
            det_file = os.path.join(
                sim_output_dir,
                self.net_file.split("/")[-1].split(".")[0] + ".det.xml",
            )
            sim_output_path = sim_output_path.split("/")[-1]
            with open(det_file, "w") as file:
                file.write("<additional>\n")
                for _, row in self.links.iterrows():
                    for i in range(0, len(row["lane"])):
                        file.write(
                            f"\t<inductionLoop" + \
                            f" id=\"{row['lane'][i]}\"" + \
                            f" lane=\"{row['lane'][i]}\"" + \
                            f" pos=\"{str(np.float64(row['length'][i]))}\"" + \
                            f" freq=\"{str(interval_increment)}\"" + \
                            f" file=\"{sim_output_path}\"/>\n"
                        )
                file.write("</additional>\n")
            self.sumo_det_all_file = det_file
            self.sumo_det_all = self.read_det_file(det_file)
        else:
            raise ValueError("TODO")

    def get_nodes_link_map(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        if self.links is not None:
            _nodes_link_map = {}
            for _, row in self.links.iterrows():
                link = row["edge"]
                _from = row["from"]
                _to = row["to"]
                if not link in self._nodes_link_map:
                    link_dict = {"link": link}
                    _nodes_link_map[(_from, _to)] = link_dict
            return _nodes_link_map
        else:
            raise ValueError("TODO")

    def determine_sensor_locations(
        self,
        sensor_coverage: float,
        ) -> Dict[Tuple[str, str], Dict[str, Any]]:
        # Return all links not in a TAZ for now
        locations = {}
        for u, v in self._nodes_link_map:
            if f"{u}_{v}" in self.links_not_in_taz:
                locations[(u, v)] = self._nodes_link_map[(u, v)]
        return locations

    def construct_det_subset_file(self, sensor_coverage: float = 1.0) -> None:
        dir_path = os.path.dirname(self.net_file)
        if dir_path == "":
            det_subset_file = self.net_file.split("/")[-1].split(".")[0] + \
                "_" + str(int(np.round(sensor_coverage * 100))) + \
                ".det_subset.xml"
        else:
            det_subset_file = dir_path + "/" + \
                self.net_file.split("/")[-1].split(".")[0] + "_" + \
                str(int(sensor_coverage * 100)) + ".det_subset.xml"
        equipped_links = self.determine_sensor_locations(
            sensor_coverage=sensor_coverage
        )
        with open(det_subset_file, "w") as file:
            file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n")
            file.write("<detSubset>\n")
            file.write("\t"*1 + "<info>\n")
            file.write("\t"*2 + "<randomSeed val=\"" + \
                str(None) + "\"" + "/>\n")
            file.write("\t"*2 + "<networkFile val=\"" + \
                str(os.path.basename(self.net_file)) + "\"" + "/>\n")
            file.write("\t"*2 + "<arcsInNetwork val=\"" + \
                str(None) + "\"" + "/>\n")
            file.write("\t"*2 + "<arcsInTAZ val=\"" + \
                str(None) + "\"" + "/>\n")
            file.write("\t"*2 + "<arcsNotInTAZ val=\"" + \
                str(None) + "\"" + "/>\n")
            file.write("\t"*2 + "<detectorsInNetwork val=\"" + \
                str(len(equipped_links)) + "\"" + "/>\n")
            # file.write("\t"*2 + "<odPairs val=\"" + \
            #     str(len(self._od_pairs)) + "\"" + "/>\n")
            file.write("\t"*1 + "</info>\n\n")
            file.write("\t"*1 + "<additional>\n")
            for u, v in equipped_links:
                file.write("\t"*2 + "<inductionLoop id=\"" + \
                self._nodes_link_map[(u, v)]["link"] + "\"" + "/>\n")
            file.write("\t"*1 + "</additional>\n")
            file.write("</detSubset>\n")

    @staticmethod
    def read_taz_file(taz_file: str) -> pd.DataFrame:
        root = ut.read_xml_file(taz_file)
        taz = []
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
        df = pd.DataFrame(data=taz, dtype=object)
        return df

    def get_link_speed(self, id_) -> float:
        if self.links is not None:
            return float(
                self.links["speed"].loc[self.links["edge"] == id_].iat[0][0]
            )
        else:
            raise ValueError("TODO")