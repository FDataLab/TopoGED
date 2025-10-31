# Models in use
from GraphGeneration.models.MultiHeadedEdgePredictor import MultiHeadedEdgePredictor
from GraphGeneration.models.temporal_gnn.script.utils.util import logger
from GraphGeneration.models.SimpleNodeLSTM import SimpleNodeLSTM
from GraphGeneration.models.GCLSTM import GCLSTM

def setupMLP(embedding_dim, mlpEncoding):   
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
    mlp = MultiHeadedEdgePredictor(in_channels=input_dim, hidden_channels=32, edge_types=flags, input_type=mlpEncoding)
    
    return mlp

def load_encoder_model(encoder_config, device, node2vec_dimensions, hidden_dim=64, num_layers=1, HTGN_nodelist=[]):       
    if encoder_config["encoder_model"]["nodeEmbeddingType"] == 'LSTM':
        model = SimpleNodeLSTM(input_dim=node2vec_dimensions, hidden_dim=hidden_dim).to(device)
    elif encoder_config["encoder_model"]["nodeEmbeddingType"] == 'GCLSTM':
        model = GCLSTM(input_dim=node2vec_dimensions, hidden_dim=hidden_dim, num_layers=num_layers, device=device).to(device)
    elif encoder_config["encoder_model"]["nodeEmbeddingType"] == 'Node2Vec':
        return None, node2vec_dimensions
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(encoder_config["encoder_model"]["nodeEmbeddingType"]))
    return model, hidden_dim
