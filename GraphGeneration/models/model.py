# Models in use
from GraphGeneration.models.MultiHeadedEdgePredictor import MultiHeadedEdgePredictor
from GraphGeneration.models.EdgePredictor import EdgePredictorMLP
from GraphGeneration.models.temporal_gnn.script.utils.util import logger
from GraphGeneration.models.temporal_gnn.script.models.HTGN import HTGN
from GraphGeneration.models.GCLSTM import GCLSTM
from GraphGeneration.models.SimpleNodeLSTM import SimpleNodeLSTM

def setupMLP(embedding_dim, strategy, embedding, mlpEncoding, embedOld, edge_vocab_size=0):   
    """
    Set up the MLP based on the arguments provided in the command line starter
    
    Args:
        None
    
    Returns:
        None
    """ 
    input_dim = embedding_dim  # Starting input dimension (two 32-dim node embeddings)
    
    # Input size changes if we are doing different methods, this keeps it consistent
    if 'Position' in embedding:
        input_dim += 2
    if 'NodeType' in embedding:
        input_dim += 2
        
    # Set up the MLPs according to arguments
    if strategy == 'SingleMLP':
        mlp = EdgePredictorMLP(in_channels=input_dim, hidden_channels=32, input_type=mlpEncoding)
        
    elif strategy == 'MultiheadedMLP':
        if embedOld == 'True':
            flags = ['o-o-bank', 'o-o-nobank', 'o-n', 'n-n']
        else:
            flags = ['o-o-nobank', 'o-n', 'n-n']
        mlp = MultiHeadedEdgePredictor(in_channels=input_dim, hidden_channels=32, edge_types=flags, input_type=mlpEncoding)
        
    return mlp

def load_encoder_model(args, device, node2vec_dimensions, HTGN_nodelist=[]):
    if args.embeddingType == 'LSTM':
        model = SimpleNodeLSTM(input_dim=node2vec_dimensions+4, hidden_dim=64).to(device)
    elif args.embeddingType == 'GCLSTM':
        model = GCLSTM(in_channels=node2vec_dimensions+4, hidden_channels=64).to(device)
        model.device = device
    elif args.embeddingType == 'HTGN':
        args.num_nodes = len(HTGN_nodelist)
        args.nfeat = node2vec_dimensions + 4
        args.nhid = 64
        args.nout = 64
        model = HTGN(args).to(device)
        model.device = device
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(args.embeddingType))
    return model
