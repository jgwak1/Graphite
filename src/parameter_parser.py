import argparse # python 3.2+
import pathlib # python 3.4+

def param_parser():

    r"""  For parsing arguments from command line. """

    parser = argparse.ArgumentParser()

    parser.add_argument('--N', 
                        nargs = 1, 
                        type = int, 
                        default = 4,
                        help = 'N for Graphite N-gram')  

    parser.add_argument('--pool', 
                        nargs = 1, 
                        type = str, 
                        choices= ["sum", "mean", "max"], 
                        default = "sum",
                        help = 'pooling method for Graphite N-gram')

    parser.add_argument('--dataset-path', 
                        nargs='?',
                        type = str,
                        default= str( pathlib.Path(__file__).parent.parent.joinpath("dataset") ),
                        help = 'processed-pickle dataset directory-path'
                        )

    parser.add_argument('--eventname-edgefeats-path', 
                        nargs='?',
                        type = str,
                        default= str( pathlib.Path(__file__).parent.parent.joinpath("dataset/EventName_EdgeFeatures.json") ),
                        help = 'event-name edge-features as json'
                        )

    parser.add_argument('--nodetype-nodefeats-path', 
                        nargs='?',
                        type = str,
                        default= str( pathlib.Path(__file__).parent.parent.joinpath("dataset/NodeType_NodeFeatures.json") ),                        
                        help = 'event-name edge-features as json'
                        )

    return parser.parse_args()