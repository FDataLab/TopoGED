import argparse
import torch
import os

parser = argparse.ArgumentParser(description='HTGN')
# 1.dataset
parser.add_argument("--dataset", type=str, required=False, default='CollegeMsg', choices=['CollegeMsg', 'mathoverflow', 'networkadex', 'networkaeternity', 'networkaion', 'networkaragon', 'networkbancor', 'networkcentra', 'networkcoindash', 'Reddit_B', 'networkcindicator', 'networkiconomi', 'networkdgd'])
parser.add_argument('--data_pt_path', type=str, default='', help='need to be modified')
parser.add_argument('--num_nodes', type=int, default=-1, help='num of nodes')
parser.add_argument('--nfeat', type=int, default=128, help='dim of input feature')
parser.add_argument('--nhid', type=int, default=16, help='dim of hidden embedding')
parser.add_argument('--nout', type=int, default=16, help='dim of output embedding')
parser.add_argument('--neg_sample', type=str, default='rnd', help='negative sampling strategy')
parser.add_argument('--num_snapshots', type=int, default=100, help='num of snapshots')

# 2.experiments
parser.add_argument('--max_epoch', type=int, default=500, help='number of epochs to train.')
parser.add_argument('--testlength', type=int, default=3, help='length for test, default:3')
parser.add_argument('--targetsnapshot', type=int, default=3, help='target snapshot we wish to predict, default:3')
parser.add_argument('--seed', type=int, default=1024, help='random seed')
parser.add_argument('--repeat', type=int, default=1, help='running times')
parser.add_argument('--patience', type=int, default=50, help='patience for early stop')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-7, help='weight for L2 loss on basic models.')
parser.add_argument('--output_folder', type=str, default='', help='need to be modified')
parser.add_argument('--use_htc', type=int, default=1, help='use htc or not, default: 1')
parser.add_argument('--use_hta', type=int, default=1, help='use hta or not, default: 1')
parser.add_argument('--debug_content', type=str, default='', help='debug_mode content')
parser.add_argument('--sampling_times', type=int, default=1, help='negative sampling times')
parser.add_argument('--log_interval', type=int, default=20, help='log interval, default: 20,[20,40,...]')
parser.add_argument('--pre_defined_feature', default=None, help='pre-defined node feature')
parser.add_argument('--save_embeddings', type=int, default=0, help='save or not, default:0')
parser.add_argument('--debug_mode', type=int, default=0, help='debug_mode, 0: normal running; 1: debugging mode')
parser.add_argument('--min_epoch', type=int, default=100, help='min epoch')

# 3.models
parser.add_argument('--model', type=str, default='HTGN', help='model name')
parser.add_argument('--manifold', type=str, default='PoincareBall', help='Hyperbolic model')
parser.add_argument('--use_gru', type=bool, default=True, help='use gru or not')
parser.add_argument('--use_hyperdecoder', type=bool, default=True, help='use hyperbolic decoder or not')
parser.add_argument('--EPS', type=float, default=1e-15, help='eps')
parser.add_argument('--nb_window', type=int, default=5, help='the length of window')
parser.add_argument('--bias', type=bool, default=True, help='use bias or not')
parser.add_argument('--trainable_feat', type=int, default=0,
                    help='using trainable feat or one-hot feat, default: none-trainable feat')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (1 - keep probability).')
parser.add_argument('--heads', type=int, default=1, help='attention heads.')
parser.add_argument('--egcn_type', type=str, default='EGCNH', help='Type of EGCN: EGCNH or EGCNO')
parser.add_argument('--curvature', type=float, default=1.0, help='curvature value')
parser.add_argument('--fixed_curvature', type=int, default=1, help='fixed (1) curvature or not (0)')
parser.add_argument('--aggregation', type=str, default='deg', help='aggregation method: [deg, att]')

parser.add_argument("--strategy", type=str, required=False, default='MultiheadedMLP', choices=['MultiheadedMLP', 'SingleMLP'], help="The type of MLP NN to use")
parser.add_argument("--embedding", type=str, required=False, default='Position', choices=['Position', 'NodeType', 'Position+NodeType', 'None'], help="Allows appending positional encodings or an integer node type onto the end of the embeddings")
parser.add_argument("--mlpEncoding", type=str, required=False, default='Concat', choices=['Concat', 'Product', 'Addition', 'Subtraction'], help="How you want to input node embeddings to the MLP")  # Product and addition lead to potential noise as we use directed graphs
parser.add_argument("--embedOld", type=str, required=False, default='True', choices=['True', 'False'], help="If you want to let the MLP predict edge type \'o-o-bank\', otherwise these edges are randomly added")
parser.add_argument("--oldDegree", type=str, required=False, default='False' ,choices=['True', 'False'], help="If you want reappearing nodes to reuse their most recent degree")
parser.add_argument("--trainingStyle", type=str, required=False, default='TrueGraphs', choices=['TrueGraphs', 'PredGraphs', 'MixedGraphs'], help="When training the MLP, decides if you use real graphs, predicted graphs (with first real as starter), or real then pred for MLP training")
parser.add_argument("--embeddingType", type=str, required=False, default='Node2Vec', choices=['Linear', 'Node2Vec', 'LSTM', 'GCLSTM', 'HTGN'], help="How nodes should be embedded. Either with Node2Vec or with a Linear mutliplication of adjacency matrix by node feature matrix")

# TopoGED mode
parser.add_argument('--use_predict_probs', action='store_true', help='Use prediction probabilities to predict the next snapshot')
parser.add_argument('--use_predict_graph_prediction', action='store_true', help='Use prediction graph description to predict the next snapshot')

args = parser.parse_args()

args.output_folder = '../data/output/log/{}/{}/'.format(args.dataset, args.model)
args.result_txt = '../data/output/results/{}_{}_result.txt'.format(args.dataset, args.model)

# open debugging mode
if args.debug_mode == 1:
    print('start debugging mode!')
    folder = '../data/output/ablation_study/{}/'.format(args.debug_content)
    args.result_txt = folder + '{}_{}_result.txt'.format(args.dataset, args.model)
    if not os.path.isdir(folder):
        os.makedirs(folder)

# update the parameters for different datasets
if args.dataset in ['enron10', 'dblp', 'uci']:
    args.testlength = 3  # using one-hot feature as input

if args.dataset in ['fbw']:  # length: 36
    args.testlength = 3
    args.trainable_feat = 1  # using trainable feature as input

if args.dataset in ['HepPh30', 'HepPh60']:  # length: 36
    args.testlength = 6
    args.trainable_feat = 1  # using trainable feature as input

if args.dataset in ['as733']:
    args.testlength = 10
    args.trainable_feat = 1  # using trainable feature as input

if args.dataset in ['wiki']:
    args.testlength = 15
    args.trainable_feat = 1  # using trainable feature as input

if args.dataset in ['disease']:
    args.testlength = 3
    args.pre_defined_feature = 1  # using pre_defined_feature as input

if args.dataset in ['disease_mc']:
    args.testlength = 3
    args.pre_defined_feature = 1  # using pre_defined_feature as input

if args.dataset in ['canVote']:
    args.testlength = 1

if args.dataset in ['LegisEdgelist']:
    args.testlength = 1

if args.dataset in ['UNtrade']:
    args.testlength = 2

if args.dataset in ['aion']:
    args.testlength = 38  # train-test split: 80-20; Total number of snapshots = 190
    args.trainable_feat = 1

if args.dataset in ['dgd']:
    args.testlength = 144  # train-test split: 80-20; Total number of snapshots = 720
    args.trainable_feat = 1

if args.dataset in ['adex']:
    args.testlength = 59  # train-test split: 80-20; Total number of snapshots = 293
    args.trainable_feat = 1

if args.dataset in ['aragon']:
    args.testlength = 67  # train-test split: 80-20; Total number of snapshots = 337
    args.trainable_feat = 1

if args.dataset in ['coindash']:
    args.testlength = 54  # train-test split: 80-20; Total number of snapshots = 268
    args.trainable_feat = 1

if args.dataset in ['iconomi']:
    args.testlength = 108  # train-test split: 80-20; Total number of snapshots = 542
    args.trainable_feat = 1

if args.dataset in ['aeternity']:
    args.testlength = 46  # Total number of snapshots = 229
    args.trainable_feat = 1

if args.dataset in ['bancor']:
    args.testlength = 66  # Total number of snapshots = 331
    args.trainable_feat = 1

if args.dataset in ['centra']:
    args.testlength = 52  # Total number of snapshots = 261
    args.trainable_feat = 1

if args.dataset in ['cindicator']:
    args.testlength = 44  # Total number of snapshots = 221
    args.trainable_feat = 1

# if args.dataset in ['CollegeMsg']:
#     args.testlength = 35  # Total number of snapshots = 177
#     args.trainable_feat = 1

# if args.dataset in ['mathoverflow']:
#     args.testlength = 37  # Total number of snapshots = 183
#     args.trainable_feat = 1