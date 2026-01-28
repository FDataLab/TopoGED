# Models in use
import numpy as np
import torch
from GraphGeneration.encoders.TGN.model.tgn import TGN
from GraphGeneration.encoders.GCN import GCN
from GraphGeneration.encoders.GAT import GATLayerImp3
from GraphGeneration.models.MultiHeadedEdgePredictor import MultiHeadedEdgePredictor, ExistMultiHeadedEdgePredictor
from GraphGeneration.models.temporal_gnn.script.utils.util import logger
from GraphGeneration.models.SimpleNodeLSTM import SimpleNodeLSTM
from GraphGeneration.encoders.GCLSTM import GCLSTM

def setupMLP(embedding_dim, mlpEncoding, model_type='default'):   
    """
    Set up the MLP based on the arguments provided in the command line starter
    
    Args:
        None
    
    Returns:
        None
    """ 
    input_dim = embedding_dim  # Starting input dimension (two 32-dim node embeddings)
        
    # Set up the MLPs according to arguments
    flags = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
    if model_type=='exist':
     mlp = ExistMultiHeadedEdgePredictor(in_channels=input_dim, hidden_channels=32, edge_types=flags, input_type=mlpEncoding)
    else:
        mlp = MultiHeadedEdgePredictor(in_channels=input_dim, hidden_channels=32, edge_types=flags, input_type=mlpEncoding)
    
    return mlp

def load_encoder_model(encoder_config, device, node2vec_dimensions, hidden_dim=64, num_layers=1, HTGN_nodelist=[], node_features=None, edge_features=None):       
    if encoder_config["encoder_model"]["nodeEmbeddingType"] == 'LSTM':
        model = SimpleNodeLSTM(input_dim=node2vec_dimensions, hidden_dim=hidden_dim).to(device)
    elif encoder_config["encoder_model"]["nodeEmbeddingType"] == 'GCLSTM':
        model = GCLSTM(
            input_dim=encoder_config["encoder_model"]["other_models"]["feature_dim"],
            hidden_dim=encoder_config["encoder_model"]["other_models"]["embedding_dim"],
            num_layers=encoder_config["encoder_model"]["other_models"]["num_layers"],
            device=device
        ).to(device)
        return model, encoder_config["encoder_model"]["other_models"]["embedding_dim"]
    elif encoder_config["encoder_model"]["nodeEmbeddingType"] == 'TGN':
        if isinstance(node_features, torch.Tensor):
            node_features = node_features.cpu().numpy()
        if isinstance(edge_features, torch.Tensor):
            node_features = edge_features.cpu().numpy()
        model = TGN(
            neighbor_finder=None,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
            use_memory=True,
            memory_dimension=encoder_config["encoder_model"]["other_models"]["embedding_dim"],
            n_neighbors=encoder_config["encoder_model"]["other_models"]["n_neighbors"]
        ).to(device)
        return model, encoder_config["encoder_model"]["other_models"]["embedding_dim"]
    elif encoder_config["encoder_model"]["nodeEmbeddingType"] == 'GAT':
        model = GATLayerImp3(
            num_in_features=encoder_config["encoder_model"]["other_models"]["feature_dim"],
            num_out_features=encoder_config["encoder_model"]["other_models"]["embedding_dim"],
            num_of_heads=encoder_config["encoder_model"]["other_models"]["num_layers"]
        ).to(device)
        return model, encoder_config["encoder_model"]["other_models"]["embedding_dim"]
    elif encoder_config["encoder_model"]["nodeEmbeddingType"] == 'GCN':
        model = GCN(
            in_dim=encoder_config["encoder_model"]["other_models"]["feature_dim"],
            hidden_dim=hidden_dim,
            out_dim=encoder_config["encoder_model"]["other_models"]["embedding_dim"]
        ).to(device)
        return model, encoder_config["encoder_model"]["other_models"]["embedding_dim"]
    elif encoder_config["encoder_model"]["nodeEmbeddingType"] == 'Node2Vec':
        return None, node2vec_dimensions
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(encoder_config["encoder_model"]["nodeEmbeddingType"]))
    return model, hidden_dim
