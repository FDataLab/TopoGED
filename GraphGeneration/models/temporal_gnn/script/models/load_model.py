from GraphGeneration.models.temporal_gnn.script.utils.util import logger
from GraphGeneration.models.temporal_gnn.script.models.EvolveGCN.EGCN import EvolveGCN
from GraphGeneration.models.temporal_gnn.script.models.DynModels import DGCN
from GraphGeneration.models.temporal_gnn.script.models.HTGN import HTGN
from GraphGeneration.models.temporal_gnn.script.models.static_baselines import VGAENet, GCNNet
from GraphGeneration.models.temporal_gnn.script.models.ROLAND_DGNN import ROLANDGNN
from torch_geometric_temporal import EvolveGCNO, GCLSTM, TGCN
from GraphGeneration.models.temporal_gnn.script.models.vgae.model import VGAE
from GraphGeneration.encoders.TGN.model.tgn import TGN
from GraphGeneration.models.temporal_gnn.script.models.TGAT import TGAN
import torch
# from GraphGeneration.models.temporal_gnn.script.models.wingnn import WinGNN
# from GraphGeneration.models.temporal_gnn.script.models.evolveGCN import EvolveGCN


def load_model(args):
    if args.model in ['GRUGCN', 'DynGCN']:
        model = DGCN(args)
    elif args.model == 'HTGN':
        model = HTGN(args)
    elif args.model == 'EvolveGCN':
        model = EvolveGCN(args).to(args.device)
    elif args.model == 'GAE':
        model = GCNNet()
    elif args.model == 'VGAE':
        # Set up the adj input
        adj = torch.eye(args.num_nodes, device=args.device).to_sparse()
        model = VGAE(adj, args.input_dim, args.hidden1_dim, args.hidden2_dim)
    elif args.model == "GCLSTM":
        model = GCLSTM(in_channels=args.nfeat, out_channels=args.nhid, K=args.chebyshev_filter)
    elif args.model == "TGCN":
        model = TGCN(in_channels=args.nfeat, out_channels=args.nhid)
    elif args.model == "ROLAND":
        model = ROLANDGNN(num_nodes=args.num_nodes,nhid=args.nhid,dropout=args.dropout,input_channel=args.nfeat, update='gru')
    # elif args.model == "WinGNN":
    #     model = WinGNN(in_features= args.nfeat, out_features=args.nhid)
    elif args.model == 'TGN':
        model = TGN(neighbor_finder=args.neighbor_finder, node_features=args.node_features, edge_features=args.edge_features, device=args.device)
    elif args.model == 'TGAT':
        model = TGN(neighbor_finder=args.neighbor_finder, node_features=args.n_feat, edge_features=args.e_feat, device=args.device)
    else:
        raise Exception('pls define the models')
    logger.info('using models {} '.format(args.model))
    return model
