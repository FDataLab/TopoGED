from script.utils.util import logger
from script.models.EvolveGCN.EGCN import EvolveGCN
from script.models.DynModels import DGCN
from script.models.HTGN import HTGN
from script.models.static_baselines import VGAENet, GCNNet
from script.models.ROLAND_DGNN import ROLANDGNN

def load_model(args):
    if args.model in ['GRUGCN', 'DynGCN']:
        model = DGCN(args)
    elif args.model == 'HTGN':
        model = HTGN(args)
    elif args.model == 'EGCN':
        model = EvolveGCN(args)
    elif args.model == 'GAE':
        model = GCNNet()
    elif args.model == 'VGAE':
        model = VGAENet()
    elif args.model == 'ROLANDGNN':
        model_dim = {
            "input_dim": args.num_nodes,
            "hidden_conv_1": 16,
            "hidden_conv_2": 16
        }
        model = ROLANDGNN(model_dim=model_dim, num_nodes=args.num_nodes)
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(args.model))
    return model
