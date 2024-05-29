import os
import pickle
from typing import List # python 3.5+

import torch
from torch_geometric.data import Data


def load_dataset(benign_data_path : str, 
                 malware_data_path : str,
                 dim_node : int,
                 dim_edge : int ) -> List[Data]:
    """ 
    Load both benign and malware graph dataset

    Args
        benign_data_path (str): path to benign data
        malware_data_path (str): path to malware data 
        dim_node (int): dimension of node features
        dim_edge (int): dimension of edge features

    Returns
        loaded dataset (list): list of benign and malware pytorch-geometric Data samples
    """
    dataprocessor = LoadGraphs()
    benign_dataset = dataprocessor.parse_all_data( benign_data_path , dim_node, dim_edge )
    malware_dataset = dataprocessor.parse_all_data( malware_data_path, dim_node, dim_edge )
    loaded_dataset = benign_dataset + malware_dataset
    print(f"+ dataset loaded #Benign = {len(benign_dataset)} | #Malware = {len(malware_dataset)} from\n\t'{benign_data_path}' and\n\t'{malware_data_path}', respectively.", flush=True)
    
    return loaded_dataset


class LoadGraphs:

    """
    Data Loader class for the graphs
    code works for a graph classification task only
    """

    def __init__(self):
        return


    def load_pickle(self, filename):
        """
        loads a pickle

        Args
           filename (str): path to file

        Returns
           data : the pickled data in list format
        """
        data = None
        with open(filename, 'rb') as fp:
            try:
                data = pickle.load(fp)
                return data
            except pickle.UnpicklingError:
                return None

    def parse_single_graph(self, file_path, num_node_attr=5, num_edge_attr=80):
        """
        parses a single graph into pytorch-geometric format
        loads the pre-processed graph data

        Args
           file_path (str): abs. file path

        Returns:
           {x, edge_list, y, edge_attr}
        """
        _name = file_path.split("/")[-1]
        data = self.load_pickle(file_path)
        if data is None:
            print(">>> pickle.UnpicklingError for sample", _name)
            return -1, -1, -1, -1, -1
        
        x, y, edge_attr, edge_list = data['x'], data['y'], data['edge_attr'], data['edge_list']
        len_x = len(x)
        len_edg = len(edge_list[0])
        # skip extremely large graphs
        if len(x) > 400000:
            print(_name, ">>> #nodes:", len(x), " #edges:", len(edge_attr), " | sample skipped!")
            return -1, -1, -1, -1, -1

        num_nodes = 10
        if len(x) < num_nodes:
            print(_name, ">>> #nodes:", len(x), " #edges:", len(edge_attr), " | sample skipped!")
            return -1, -1, -1, -1, -1
        
        # check for num node attributes
        for _x in x:
            if len(_x) != num_node_attr:
                print(_name, ">>> #node attributes are mismatched, ", len(_x), ' | sample skipped!')
                return -1, -1, -1, -1, -1

        # check for num edge attributes
        for _y in edge_attr:
            if len(_y) != num_edge_attr:
                print(_name, ">>>> #edge attributes mismatched, ", len(_y), " | sample skipped!")
                return -1, -1, -1, -1, -1

        # convert all data into tensors
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        edge_list = torch.tensor(edge_list, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        print("> ", _name, "| #node: ", len_x, " #edge: ", len_edg)

        return x, edge_list, y, edge_attr, _name


    def parse_all_data(self, load_path, num_node_attr, num_edge_attr ):
        """
        Parses all data samples in 'load_path'

        Args
           load_path (str): the path to load all pickled samples

        Returns
           dataset (list): list of pytorch-geometric Data samples
        """
        dir_contents = os.listdir(load_path)
        
        dataset = []  # will store the datset here
        # loop through all samples and parse
        for idx, filename in enumerate(dir_contents):

            if '_Sample_' not in filename and '_SUBGRAPH_' not in filename:
                continue
            
            _path = load_path + "/" + filename  # path to the pickled file
            
            x, edge_list, y, edge_attr, name = self.parse_single_graph(_path, 
                                                                        num_node_attr=num_node_attr, 
                                                                        num_edge_attr=num_edge_attr)

            if isinstance(x, int) and x == -1:
                continue
            dataset.append( Data(x=x, edge_index=edge_list, edge_attr=edge_attr, y=y, name=name) )



        return dataset

