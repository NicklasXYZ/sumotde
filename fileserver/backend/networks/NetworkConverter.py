from multiprocessing import Value
from typing import Dict, Any
import tempfile
import logging

import os
import numpy as np

# from backend.networks.CommandLineArgs import CommandLineArgs
import backend.networks.networks as networks 
import backend.utils as ut


class NetworkConverter:

    def __init__(
        self,
        cmd_args: Dict[str, Any],
        network_data: networks.BaseNetwork,
        to_network: str = "sumo"
        ) -> None:
        self._cmd_args = cmd_args
        self._network_data = network_data
        self._file_name = network_data.name
        self._to_network = to_network
        # Give the output file the approriate name (for identification purposes)
        if self._file_name.split("-")[-1] == "od":
            self._file_name += str(len(network_data._od_pairs))
        # Build a sumo network
        if to_network == "sumo":
            self._to_sumo_network()
            self._construct_sumo_taz_file()
            # TODO: Make sure 'interval_increment' is contained in the dict
            # if self._cmd_args["interval_increment"] is None:
            #     interval_increment = None
            # else:
            #     interval_increment = self._cmd_args["interval_increment"]
            # self._network_data.generate_det_file(
            #     sim_output_path=self._cmd_args["output_dir"],
            #     sim_output_dir=self._cmd_args["output_dir"],
            #     interval_increment=interval_increment,
            # )
            # quit()
        # Build a matsim network
        elif to_network == "matsim":
            self._to_matsim_network()
            self._construct_sumo_taz_file()
        else:
            logging.error(f"Wrong network type specified: {self._to_network}")
            logging.error(
                f"Specify whether we should convert the Networkx MultiDiGraph \
                object to \"sumo\" or a \"matsim\" traffic network!"
            )
            raise ValueError("TODO")
    
    @property
    def network_file(self):
        if self._cmd_args is not None and self._file_name is not None:
            network_file =os.path.join(
                self._cmd_args["output_dir"],
                self._file_name + ".net.xml"
            )
            return network_file
        else:
            raise ValueError("TODO")

    def _to_sumo_network(self):
        link_file = self._write_sumo_links_to_file()
        logging.info(f"Creating temporary linkfile: {link_file.name}")
        node_file = self._write_sumo_nodes_to_file()
        logging.info(f"Creating temporary nodefile: {node_file.name}")
        
        if not os.path.exists(self._cmd_args["output_dir"]):
            print(f"Creating directory: {self._cmd_args['output_dir']}")
            os.makedirs(self._cmd_args["output_dir"])

        commandline_args = "netconvert" + " --verbose true " + \
            " --node-files " + node_file.name + \
            " --edge-files " + link_file.name + \
            " --output-file " + self.network_file
        ut.run_subprocess(commandline_args=commandline_args)
        # Remove the temporary files
        os.remove(node_file.name)
        os.remove(link_file.name)

    def _write_sumo_links_to_file(self, default_speed: float = 277.777778):
        link_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        network = self._network_data.get_network()
        print("<edges>", file=link_file)
        for u, v, data in network.edges(data = True):
            if u in self._network_data._origins + self._network_data._destinations or \
                v in self._network_data._origins + self._network_data._destinations:
                print(4 * "\t" + \
                    "<edge id=\"%s\" from=\"%s\" to=\"%s\" numLanes=\"%s\" speed=\"%s\">" % \
                    (str(u) + "_" + str(v), u, v, data["lanes"], str(default_speed)),
                    file=link_file,
                )
            else:
                print(4 * "\t" + \
                    "<edge id=\"%s\" from=\"%s\" to=\"%s\" numLanes=\"%s\" speed=\"%s\">" % \
                    (str(u) + "_" + str(v), u, v, data["lanes"], data["maxspeed"]),
                    file=link_file,
                )
            print(4 * "\t" + "</edge>", file=link_file)
        print("</edges>", file=link_file)
        link_file.close()
        return link_file

    def _write_sumo_nodes_to_file(self):
        nodes_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
        network = self._network_data.get_network()
        print("<nodes>", file=nodes_file)
        for node, data in network.nodes(data = True):
            print(4 * "\t" + \
                "<node id=\"%s\" x=\"%s\" y=\"%s\"/>" % \
                (node, data["x"], data["y"]),
                file=nodes_file
            )
        print("</nodes>", file=nodes_file)
        nodes_file.close()
        return nodes_file

    def _to_matsim_network(self):
        raise NotImplementedError()

    def _write_matsim_links_to_file(self):
        raise NotImplementedError()

    def _write_matsim_nodes_to_file(self):
        raise NotImplementedError()

    def _get_offset(self):
        net_file = os.path.join(
            self._cmd_args["output_dir"],
            self._file_name + ".net.xml"
        )
        logging.info(f"Getting offset from network file: {net_file}")
        root = ut.read_xml_file(net_file)
        for child in root:
            if child.tag == "location":
                if "netOffset" in child.attrib:
                    offset = child.attrib["netOffset"].split(",")
                    offset_x = float(offset[0])
                    offset_y = float(offset[1])
                else:
                    offset_x = 0.0
                    offset_y = 0.0
        return offset_x, offset_y

    def _set_taz(self):
        offset_x, offset_y = self._get_offset()
        logging.info(f"TAZ coordinate system offset. X-OFFSET {offset_x} Y_OFFSET {offset_y}")
        # Assume origins and destinations are centroids
        # We then simply make the TAZ around the centroid
        # Loop over origin centroids which serve as the middle point of a TAZ
        scale = [0, 0]; counter = 0
        for node, data in self._network_data._network.nodes(data=True):
            scale[0] += data["x"] + offset_x; scale[1] += data["y"] + offset_y; counter += 1
        scale[0] /= counter; scale[1] /= counter
        taz = {}; counter = 0; nodes = []
        for node in self._network_data._origins + self._network_data._destinations:
            x = self._network_data._network.nodes[str(node)]["x"] + offset_x
            y = self._network_data._network.nodes[str(node)]["y"] + offset_y
            x1 = x + scale[0] * 0.0025
            x2 = x - scale[0] * 0.0025
            y1 = y + scale[1] * 0.0025
            y2 = y - scale[1] * 0.0025
            if not node in nodes:
                if not "taz" + "_" + str(counter) in taz:
                    taz["taz" + "_" + str(counter)] = {
                        "sources": [],
                        "sinks": [],
                        "coords": None
                    }
                for link in self._network_data._network.out_edges(str(node)):
                    link_id = str(link[0]) + "_" + str(link[1])
                    # Link is a source:
                    if str(node) in self._network_data._origins:
                        taz["taz" + "_" + str(counter)]["sources"].append(link_id)
                        taz["taz" + "_" + str(counter)]["coords"] = \
                            str(x1) + "," + str(y1) + " " + \
                            str(x1) + "," + str(y2) + " " + \
                            str(x2) + "," + str(y2) + " " + \
                            str(x2) + "," + str(y1) + " " + \
                            str(x1) + "," + str(y1)
                for link in self._network_data._network.in_edges(str(node)):
                    link_id = str(link[0]) + "_" + str(link[1])
                    # Link is a sink:
                    if str(node) in self._network_data._destinations:
                        taz["taz" + "_" + str(counter)]["sinks"].append(link_id)
                        taz["taz" + "_" + str(counter)]["coords"] = \
                            str(x1) + "," + str(y1) + " " + \
                            str(x1) + "," + str(y2) + " " + \
                            str(x2) + "," + str(y2) + " " + \
                            str(x2) + "," + str(y1) + " " + \
                            str(x1) + "," + str(y1)
                nodes.append(str(node))
                counter += 1
        return taz

    def _source_tag_content(self, f, link_id):
        f.write(2 * "\t" + "<tazSource weight=" + "\"" + str(1) + "\"" + " " + \
            "id=" + "\"" + link_id + "\"" + "/>\n")

    def _sink_tag_content(self, f,    link_id):
        f.write(2 * "\t" + "<tazSink weight=" + "\"" + str(1) + "\"" + " " + \
            "id=" + "\"" + link_id + "\"" + "/>\n")

    def _taz_tag(self, f, taz_id, shape, source_links, sink_links):
        if len(source_links) != 0 or len(sink_links) != 0:
            f.write("\t" + "<taz id=" + "\"" + taz_id + "\"" + " " + \
                "shape=" + "\"" + shape + "\"" + ">\n")
            if len(source_links) != 0:
                for link_id in source_links:
                    self._source_tag_content(f, link_id)
            if len(sink_links) != 0:
                for link_id in sink_links:
                    self._sink_tag_content(f, link_id)
            f.write("\t" + "</taz>\n")

    @property
    def taz_file(self):
        if self._cmd_args is not None and self._file_name is not None:
            taz_file = os.path.join(
                self._cmd_args["output_dir"],
                self._file_name + ".taz.xml"
            )
            return taz_file
        else:
            raise ValueError("TODO")

    def _construct_sumo_taz_file(self):
        taz = self._set_taz()
        with open(file=self.taz_file, mode="w") as file:
            file.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\n")
            file.write(
                "<additional xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"" + \
                " " + \
                "xsi:noNamespaceSchemaLocation=\"http://sumo.dlr.de/xsd/additional_file.xsd\">" + \
                "\n"
            )
            for item in taz:
                self._taz_tag(
                f=file,
                taz_id=item,
                shape=taz[item]["coords"],
                source_links=taz[item]["sources"],
                sink_links=taz[item]["sinks"]
            )
            file.write("</additional>\n")