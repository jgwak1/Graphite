from parameter_parser import param_parser
from dataprocessor_graphs import load_dataset
from graphite_n_gram import Graphite_Ngram

import os
import json
from sklearn.metrics import accuracy_score, f1_score 

def main(args):

    eventname_edgefeats = json.load( open( args.eventname_edgefeats_path, "r") )
    nodetype_nodefeats = json.load( open( args.nodetype_nodefeats_path, "r") )
    train_dataset = load_dataset( benign_data_path = os.path.join( args.dataset_path, "train/benign"),  malware_data_path = os.path.join( args.dataset_path, "train/malware"), dim_node = len(nodetype_nodefeats),  dim_edge= len(eventname_edgefeats) + 1 ) # +1 for event-timestamp
    test_dataset = load_dataset( benign_data_path =  os.path.join( args.dataset_path,"test/benign"), malware_data_path =  os.path.join( args.dataset_path,"test/malware"), dim_node = len(nodetype_nodefeats), dim_edge= len(eventname_edgefeats) + 1 )  # +1 for event-timestamp

    graphite_ngram = Graphite_Ngram( N = args.N, pool= args.pool )

    graphite_ngram.fit( train_dataset = train_dataset,  nodetype_nodefeats = nodetype_nodefeats,  eventname_edgefeats= eventname_edgefeats )

    preds, truths = [], []
    for test_data in test_dataset:
        pred = graphite_ngram.predict( test_data )
        truth  = [ 1 if "malware" in test_data.name else 0 ][0]
        print(f"Predicted: { pred } | Truth: {truth}   ---   {test_data.name}", flush=True)
        preds.append(pred)
        truths.append(truth)


    test_acc = accuracy_score(y_true = truths, y_pred = preds)
    test_f1 = f1_score(y_true = truths, y_pred = preds)

    print("-"*50, flush=True)
    print(f"Test-Acc: {test_acc} | Test-F1 : {test_f1}", flush=True)


    return


if __name__ == "__main__":
    args = param_parser()
    main(args)