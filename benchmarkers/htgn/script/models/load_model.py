from benchmarkers.htgn.script.utils.util import logger
from benchmarkers.htgn.script.models.EvolveGCN.EGCN import EvolveGCN
from benchmarkers.htgn.script.models.DynModels import DGCN
from benchmarkers.htgn.script.models.HTGN import HTGN
from benchmarkers.htgn.script.models.static_baselines import VGAENet, GCNNet


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
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(args.model))
    return model
