# Models in use
import os
import torch
from GraphGeneration.models.MultiHeadedEdgePredictor import MultiHeadedEdgePredictor
from GraphGeneration.models.EdgePredictor import EdgePredictorMLP
from GraphGeneration.models.temporal_gnn.script.utils.util import logger
from GraphGeneration.models.temporal_gnn.script.models.HTGN import HTGN
from GraphGeneration.models.GCLSTM import GCLSTM
from GraphGeneration.models.SimpleNodeLSTM import SimpleNodeLSTM

def setupMLP(embedding_dim, embedding, mlpEncoding, embedOld):   
    """
    Set up the MLP based on the arguments provided in the command line starter
    
    Args:
        None
    
    Returns:
        None
    """ 
    input_dim = embedding_dim  # Starting input dimension (two 32-dim node embeddings)
        
    # Set up the MLPs according to arguments
    if embedOld == 'True':
        flags = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
    else:
        flags = ['o-o-nobank', 'o-n', 'n-n']
    mlp = MultiHeadedEdgePredictor(in_channels=input_dim, hidden_channels=32, edge_types=flags, input_type=mlpEncoding)
    
    return mlp

def load_encoder_model(args, device, node2vec_dimensions, hidden_dim=64, HTGN_nodelist=[]):
    # Input size changes if we are doing different methods, this keeps it consistent
    if args.embedding in ['Position', 'NodeType']:
        hidden_dim += 1
        
    if args.embeddingType == 'LSTM':
        model = SimpleNodeLSTM(input_dim=node2vec_dimensions, hidden_dim=hidden_dim).to(device)
    elif args.embeddingType == 'GCLSTM':
        model = GCLSTM(in_channels=node2vec_dimensions, hidden_channels=64).to(device)
        model.device = device
    elif args.embeddingType == 'HTGN':
        args.num_nodes = len(HTGN_nodelist)
        args.nfeat = node2vec_dimensions
        args.nhid = 64
        args.nout = 64
        model = HTGN(args).to(device)
        model.device = device
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(args.embeddingType))
    return model, hidden_dim
