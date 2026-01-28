from script.utils.util import logger
from script.models.EvolveGCN.EGCN import EvolveGCN
from script.models.DynModels import DGCN
from script.models.HTGN import HTGN
from script.models.static_baselines import VGAENet, GCNNet
from script.models.ROLAND_DGNN import ROLANDGNN
from torch_geometric_temporal import EvolveGCNO, GCLSTM, TGCN
from script.models.wingnn import WinGNN
from script.models.evolveGCN import EvolveGCN

def load_model(args):
    if args.model in ['GRUGCN', 'DynGCN']:
        model = DGCN(args)
    elif args.model == 'HTGN':
        model = HTGN(args)
    elif args.model == 'EvolveGCN':
        model = EvolveGCN(node_feat_dim=args.nfeat, hidden_dim=args.nhid).to(args.device)
    elif args.model == 'GAE':
        model = GCNNet()
    elif args.model == 'VGAE':
        model = VGAENet()
    elif args.model == "GCLSTM":
        model = GCLSTM(in_channels=args.nfeat, out_channels=args.nhid, K=args.chebyshev_filter)
    elif args.model == "TGCN":
        model = TGCN(in_channels=args.nfeat, out_channels=args.nhid)
    elif args.model == "ROLAND":
        model = ROLANDGNN(num_nodes=args.num_nodes,nhid=args.nfeat,dropout=args.dropout,input_channel=args.nfeat)
    elif args.model == "WinGNN":
        model = WinGNN(in_features= args.nfeat, out_features=args.nhid)
        
    else:
        raise Exception('pls define the models')
    logger.info('using models {} '.format(args.model))
    return model
