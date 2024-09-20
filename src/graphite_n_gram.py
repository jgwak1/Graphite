from typing import List # python 3.5+

import torch
from torch_geometric.data import Data
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier


class Graphite_Ngram:
   r""" An implementation of Graphite N-gram. 
   
   Args:
      N (int) : for N-gram features
      pool (str) : pooling method 
   """

   def __init__(self, 
                N : int = 4, 
                pool : str = "sum"):
      
      self.N = N
      pool_choices = {"sum": torch.sum, "mean": torch.mean, "max": torch.max}
      self.pool = pool_choices[ pool ]
      self.count_vectorizer = CountVectorizer( ngram_range = (N, N), 
                                               max_df = 1.0, 
                                               min_df = 1, 
                                               max_features = None)

      # Tuned hyperparameters based on 10-fold cross-validation on the training dataset
      self.base_model = RandomForestClassifier(
         n_estimators= 500,
         criterion= 'gini', 
         max_depth= 20,
         min_samples_split= 2, 
         min_samples_leaf= 1, 
         max_features= 'sqrt',
         bootstrap= False,
         random_state= 42
      )
      return


   def _get_thread_sorted_event_sequence(self,
                                         data : Data, 
                                         thread_node_idx : int ) -> List[str]:
      
      r"""
      Extract thread-level event-sequence of a single-data
      """

      # get thread's incoming and outgoing edges
      edge_src_node_indices = data.edge_index[0]
      edge_tar_node_indices = data.edge_index[1]
      outgoing_edges_from_thread = torch.nonzero( edge_src_node_indices == thread_node_idx ).flatten() # edge-indices of outgoing-edges from thread-node
      incoming_edges_to_thread = torch.nonzero( edge_tar_node_indices == thread_node_idx ).flatten() # edge-indices of incoming-edges to thread-node
      
      # get edge-feature-vectors of thread's incoming and outgoing edges
      edge_feats_of_outgoing_edges_from_thread = data.edge_attr[ outgoing_edges_from_thread ]
      edge_feats_of_incoming_edges_to_thread = data.edge_attr[ incoming_edges_to_thread ]
      edge_feats_of_all_edges_of_thread = torch.cat([edge_feats_of_incoming_edges_to_thread, edge_feats_of_outgoing_edges_from_thread], dim = 0) 
      
      # sort by timestamp
      sort_by_timestamp = torch.argsort( edge_feats_of_all_edges_of_thread[:, -1], descending=False ) 
      edge_feats_of_all_edges_of_thread__sorted = edge_feats_of_all_edges_of_thread[ sort_by_timestamp ]

      # get thread's event-sequence sorted by timestamp
      eventname_indices = torch.nonzero( edge_feats_of_all_edges_of_thread__sorted[:,:-1], as_tuple=False)[:, -1]
      thread_sorted_event_sequence = [ self.eventname_edgefeats[i] for i in eventname_indices ]
      
      return thread_sorted_event_sequence


   def _get_thread_neighboring_nodetypes(self,
                                         data : Data, 
                                         thread_node_idx : int ) -> torch.tensor:
      
      r""" Extract node-type distribution of thread's neighboring nodes """
      
      # get thread's incoming and outgoing edges
      edge_src_node_indices = data.edge_index[0]
      edge_tar_node_indices = data.edge_index[1]
      outgoing_edges_from_thread = torch.nonzero( edge_src_node_indices == thread_node_idx ).flatten() # edge-indices of outgoing-edges from thread-node
      incoming_edges_to_thread = torch.nonzero( edge_tar_node_indices == thread_node_idx ).flatten() # edge-indices of incoming-edges to thread-node
      
      # get src/dst of thread's incoming and outgoing edges, while dropping duplicate edges
      src_dst_of_outgoing_edges_from_thread, _ = torch.unique( data.edge_index[:, outgoing_edges_from_thread ], dim=1, return_inverse=True)
      src_dst_of_incoming_edges_to_thread, _ = torch.unique( data.edge_index[:, incoming_edges_to_thread ], dim=1, return_inverse=True)

      # get thread's neighbors connected by incoming and outgoing edges, while dropping duplicate nodes
      dst_of_outgoing_edges_from_thread = src_dst_of_outgoing_edges_from_thread[1] # edge-dst is at index 1
      src_of_incoming_edges_to_thread = src_dst_of_incoming_edges_to_thread[0] # edge-src is at index 0
      thread_neighboring_nodes = torch.unique( torch.cat( [ dst_of_outgoing_edges_from_thread, src_of_incoming_edges_to_thread ] ) )
 
      # get thread neighboring node-types
      nodetype_featvects = data.x
      thread_neighboring_nodetypes = torch.sum( nodetype_featvects[ thread_neighboring_nodes ], dim = 0 ).view(1,-1)

      return thread_neighboring_nodetypes


   def fit_count_vectorizer(self, train_dataset : List[Data]) -> None:
      r"""
      Extract all thread-level event-sequences from training-graphs,
      then train the N-gram count-vectorizer on them
      """

      thread_nodetype = torch.tensor([1 if _type.lower() == "thread" else 0 for _type in self.nodetype_nodefeats])
      all_thread_level_event_sequences = []

      cnt = 1
      for train_data in train_dataset:            
            
         nodetype_featvects = train_data.x
         thread_node_indices = torch.nonzero( torch.all( torch.eq( nodetype_featvects, thread_nodetype ), dim=1 ), as_tuple=False).flatten()

         for thread_node_idx in thread_node_indices.tolist():
            thread_sorted_event_sequence = self._get_thread_sorted_event_sequence( data = train_data, 
                                                                                   thread_node_idx = thread_node_idx)
            all_thread_level_event_sequences.append( thread_sorted_event_sequence )
         
         print(f"{cnt} / {len(train_dataset)}: {train_data.name} -- extracted thread-level event-sequences", flush = True)
         cnt += 1
   
      all_thread_level_event_sequences_for_fitting = [ ' '.join(thread_event_seq) for thread_event_seq in all_thread_level_event_sequences if len(thread_event_seq) >= self.N ] 
      self.count_vectorizer.fit( all_thread_level_event_sequences_for_fitting )
      print(f"fitted {self.N}-gram count-vectorizer on all thread-level event-sequences", flush = True)
      return 


   def generate_graph_embedding(self, 
                                data : Data) -> torch.tensor:
      r"""
      Generate graph-embedding for a single data
      """
      thread_nodetype = torch.tensor([1 if _type.lower() == "thread" else 0 for _type in self.nodetype_nodefeats])
      nodetype_featvects = data.x
      thread_node_indices = torch.nonzero( torch.all(torch.eq( nodetype_featvects, thread_nodetype ), dim=1), as_tuple=False).flatten()


      all_thread_node_embeddings = torch.tensor([]) # go with this 

      for thread_node_idx in thread_node_indices.tolist():
         
         # get thread-node-embedding's N-gram component 
         thread_sorted_event_sequence = self._get_thread_sorted_event_sequence( data = data, 
                                                                                thread_node_idx = thread_node_idx )
         thread_sorted_event_sequence_for_transform = " ".join(thread_sorted_event_sequence)
         thread_Ngram_count_vector = self.count_vectorizer.transform( [ thread_sorted_event_sequence_for_transform ] ).toarray()
         thread_Ngram_count_tensor = torch.Tensor( thread_Ngram_count_vector ).view(1,-1)

         # get thread-node-embedding's neighboring node-type component 
         thread_neighboring_nodetypes_tensor = self._get_thread_neighboring_nodetypes( data = data, 
                                                                                       thread_node_idx = thread_node_idx )

         thread_node_embedding = torch.cat( [thread_neighboring_nodetypes_tensor, thread_Ngram_count_tensor], dim = 1)

         all_thread_node_embeddings = torch.cat( ( all_thread_node_embeddings, thread_node_embedding ) , dim = 0 ) # append

      graph_embedding = self.pool(all_thread_node_embeddings, dim = 0)            

      return graph_embedding




   def fit(self, train_dataset : List[Data],
                 nodetype_nodefeats : List[str],
                 eventname_edgefeats : List[str],
                  ) -> None:
      
      r"""
      Fits the N-gram CountVectorizer and utilizes it to generate graph embeddings of the train dataset, then fits the base model.
      """
      
      self.nodetype_nodefeats, self.eventname_edgefeats = nodetype_nodefeats, eventname_edgefeats
      self.fit_count_vectorizer( train_dataset )

      train_data_dict = dict()
      cnt = 1
      for train_data in train_dataset:
         print(f"{cnt} / {len(train_dataset)}: {train_data.name} -- generate graph-embedding", flush = True)
         train_data_graph_embedding = self.generate_graph_embedding( train_data )
         train_data_dict[ train_data.name ] = train_data_graph_embedding.tolist()
         cnt+=1

      X = list( train_data_dict.values() )
      y = [ 1 if "malware" in data_name else 0 for data_name in train_data_dict.keys() ]
      self.base_model.fit(X = X, y = y)
      print(f"fitted base-model on train dataset", flush = True)
      return   






   def predict(self, test_data : Data):
      r"""
      Makes a prediction on a single test data

      Args
         test_data (Data)
      
      Returns
         predicted-label (int) : malware: 1, benign: 0 

      """
      test_data_graph_embedding = self.generate_graph_embedding( test_data )
      return self.base_model.predict( [ test_data_graph_embedding.tolist() ] ).item()


