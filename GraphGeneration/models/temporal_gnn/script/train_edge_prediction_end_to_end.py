"""
Assumption:
    Train and test temporal link prediction task 
    without having a pre-trained model

July 14, 2025
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from math import isnan
from sklearn.metrics import roc_auc_score, average_precision_score
from pickle import dump, load
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
print("Added to sys.path:", path)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
  

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, input_type='Concat'):
        super().__init__()

        self.input_type = input_type
        self.heads = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1),
                nn.Sigmoid()
           )

    def forward(self, src_embed, dst_embed):
        if self.input_type == 'Concat':
            edge_input = torch.cat([src_embed, dst_embed], dim=1)
        elif self.input_type == 'Addition':
            edge_input = src_embed + dst_embed
        elif self.input_type == 'Subtraction':
            edge_input = src_embed - dst_embed
        elif self.input_type == 'ElementwiseProduct':
            edge_input = src_embed * dst_embed
            
        return self.heads(edge_input).squeeze()

def extra_dataset_attributes_loading(args, readout_scheme='mean'):
    """
    Load and process additional dataset attributes for TG-Classification
    This includes graph labels and node features for the nodes of each snapshot
    """
    partial_path = f'../data/input/raw/{args.dataset}/'
   
    # load graph lables
    label_filename = f'{partial_path}/{args.dataset}_labels.csv'
    label_df = pd.read_csv(label_filename, header=None, names=['label'])
    TG_labels = torch.from_numpy(np.array(label_df['label'].tolist())).to(args.device)

    # load and process graph-pooled (node-level) features 
    edgelist_filename = f'{partial_path}/{args.dataset}_edgelist.txt'
    edgelist_df = pd.read_csv(edgelist_filename)
    uniq_ts_list = np.unique(edgelist_df['snapshot'])
    TG_feats = []
    for ts in uniq_ts_list:
       ts_edges = edgelist_df.loc[edgelist_df['snapshot'] == ts, ['source', 'destination', 'weight']]
       ts_G = nx.from_pandas_edgelist(ts_edges, source='source', target='destination', edge_attr='weight', create_using=nx.MultiDiGraph)
       node_list = list(ts_G.nodes)
       indegree_list = np.array(ts_G.in_degree(node_list))
       weighted_indegree_list = np.array(ts_G.in_degree(node_list, weight='weight'))
       outdegree_list = np.array(ts_G.out_degree(node_list))
       weighted_outdegree_list = np.array(ts_G.out_degree(node_list, weight='weight'))

       if readout_scheme == 'max':
        TG_this_ts_feat = np.array([np.max(indegree_list), np.max(weighted_indegree_list), 
                                    np.max(outdegree_list), np.max(weighted_outdegree_list)])
       elif readout_scheme == 'mean':
        TG_this_ts_feat = np.array([np.mean(indegree_list), np.mean(weighted_indegree_list), 
                                    np.mean(outdegree_list), np.mean(weighted_outdegree_list)])
       elif readout_scheme == 'sum':
        TG_this_ts_feat = np.array([np.sum(indegree_list), np.sum(weighted_indegree_list), 
                                    np.sum(outdegree_list), np.sum(weighted_outdegree_list)])
       else:
        TG_this_ts_feat = None
        raise ValueError("Readout scheme is Undefined!")
       
       TG_feats.append(TG_this_ts_feat)
    
    # scale the temporal graph features to have a reasonable range
    scalar = MinMaxScaler()
    TG_feats = scalar.fit_transform(TG_feats)

    return TG_labels, TG_feats
  

class Runner(object):
    def __init__(self):
        self.readout_scheme = 'mean'
        self.tgc_lr = 1e-4

        self.len = data['time_length']
        self.start_train = 0
        self.train_shots = list(range(self.start_train, self.len - args.testlength))
        self.test_shots = list(range(self.len - args.testlength, self.len))
        self.load_feature()
        logger.info('INFO: total length: {}, train length: {}, test length: {}'.format(self.len, len(self.train_shots), args.testlength))

        self.model = load_model(args).to(args.device)
        # self.model_path = '../saved_models/{}/{}_{}_seed_{}.pth'.format(args.dataset, args.dataset,
        #                                                            args.model, args.seed)
        # logger.info("The model is going to be loaded from {}".format(self.model_path))
        # self.model.load_state_dict(torch.load(self.model_path))

        # load the graph labels
        # self.t_graph_labels, self.t_graph_feat = extra_dataset_attributes_loading(args)

        # define decoder: graph classifier
        num_extra_feat = 4  # = len([in-degree, weighted-in-degree, out-degree, weighted-out-degree])
        self.tgc_decoder = MLP(in_channels=args.nout*2)  # @NOTE: these hyperparameters may need to be changed 

    def load_feature(self):
        if args.trainable_feat:
            self.x = None
            logger.info("INFO: Using trainable feature, feature dim: {}".format(args.nfeat))
        else:
            if args.pre_defined_feature is not None:
                import scipy.sparse as sp
                if args.dataset == 'disease':
                    feature = sp.load_npz(disease_path).toarray()
                self.x = torch.from_numpy(feature).float().to(args.device)
                logger.info('INFO: using pre-defined feature')
            else:
                self.x = torch.eye(args.num_nodes).to(args.device)
                logger.info('INFO: using one-hot feature')
            args.nfeat = self.x.size(1)


    def tgclassification_test(self, epoch):
        """
        Final inference on the test set
        """
        tg_labels, tg_preds = [], []
        for t_test_idx, t in enumerate(self.test_shots[:1]):
           self.model.eval()
           self.tgc_decoder.eval()
           with torch.no_grad():
                edge_index, pos_index, neg_index, node_list, edge_weight, _, _, node_id_map = prepare(data, t)
                embeddings, x_embeddings = self.model(edge_index, x=self.x, node_id_list=node_list, node_id_map=node_id_map)
                
                # 1. Stack edges: shape [2, N_total]
                all_edges = torch.cat([pos_index, neg_index], dim=1)
                
                # 2. Create labels: shape [N_total]
                pos_labels = torch.ones(pos_index.shape[1], dtype=torch.float32)
                neg_labels = torch.zeros(neg_index.shape[1], dtype=torch.float32)
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)

                # 3. Shuffle the edges and labels in unison
                perm = torch.randperm(all_edges.shape[1])
                shuffled_edges = all_edges[:, perm]
                shuffled_labels = all_labels[perm].float()
                
                # Remap edge indices using node_id_map
                src_nodes = shuffled_edges[0].tolist()
                dst_nodes = shuffled_edges[1].tolist()

                # Map raw node IDs → indices in embeddings using node_id_map
                try:
                    mapped_src = torch.tensor([node_id_map[int(n)] for n in src_nodes], dtype=torch.long).to(args.device)
                    mapped_dst = torch.tensor([node_id_map[int(n)] for n in dst_nodes], dtype=torch.long).to(args.device)
                except KeyError as e:
                    print(f"KeyError: Node {e} not found in node_id_map.")
                    continue  # Skip this batch

                # Decode edges
                src_embeddings = torch.stack([embeddings[int(n)] for n in mapped_src])
                dst_embeddings = torch.stack([embeddings[int(n)] for n in mapped_dst])

                tg_pred = self.tgc_decoder(src_embeddings, dst_embeddings).squeeze()

                # graph classification
                tg_labels.append(shuffled_labels.detach().cpu().numpy())
                tg_preds.append(tg_pred.detach().cpu().numpy())

        all_labels = np.concatenate(tg_labels)
        all_preds = np.concatenate(tg_preds)

        # Now compute AUC and AP
        auc = roc_auc_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_preds)
        return epoch, auc, ap
        

    def run(self):
        """
        Run the temporal graph classification task
        """
        # define optimizer and criterion
        optimizer = torch.optim.Adam(
            set(self.tgc_decoder.parameters()) | set(self.model.parameters()),
            lr=self.tgc_lr
        )
        criterion = torch.nn.BCELoss()

        # load the TG-model
        self.model.init_hiddens()
        logger.info("Start training the temporal graph classification model.")

        # make sure to have the right device setup
        self.tgc_decoder = self.tgc_decoder.to(args.device)
        self.model = self.model.to(args.device)

        self.model = self.model.train()
        self.tgc_decoder = self.tgc_decoder.train()
        
        t_total_start = time.time()
        min_loss = 10
        train_avg_epoch_loss_dict = {}
        for epoch in range(1, args.max_epoch + 1):
            t_epoch_start = time.time()
            epoch_losses = []
            for t_train_idx, t_train in enumerate(self.train_shots):
                optimizer.zero_grad()
                # edge_index, pos_index, neg_index, node_list, weights, new_pos_index, new_neg_index, node_id_map
                edge_index, pos_index, neg_index, node_list, edge_weight, _, _, node_id_map = prepare(data, t_train)
                embeddings, x_embeddings = self.model(edge_index, x=self.x, node_id_list=node_list, node_id_map=node_id_map)
                
                # 1. Stack edges: shape [2, N_total]
                all_edges = torch.cat([pos_index, neg_index], dim=1)
                
                # 2. Create labels: shape [N_total]
                pos_labels = torch.ones(pos_index.shape[1], dtype=torch.float32)
                neg_labels = torch.zeros(neg_index.shape[1], dtype=torch.float32)
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)

                # 3. Shuffle the edges and labels in unison
                perm = torch.randperm(all_edges.shape[1])
                shuffled_edges = all_edges[:, perm]
                shuffled_labels = all_labels[perm].float()
                
                # Remap edge indices using node_id_map
                src_nodes = shuffled_edges[0].tolist()
                dst_nodes = shuffled_edges[1].tolist()

                # Map raw node IDs → indices in embeddings using node_id_map
                try:
                    mapped_src = torch.tensor([node_id_map[int(n)] for n in src_nodes], dtype=torch.long).to(args.device)
                    mapped_dst = torch.tensor([node_id_map[int(n)] for n in dst_nodes], dtype=torch.long).to(args.device)
                except KeyError as e:
                    print(f"KeyError: Node {e} not found in node_id_map.")
                    continue  # Skip this batch

                # Decode edges
                src_embeddings = torch.stack([embeddings[int(n)] for n in mapped_src])
                dst_embeddings = torch.stack([embeddings[int(n)] for n in mapped_dst])

                tg_pred = self.tgc_decoder(src_embeddings, dst_embeddings).squeeze()


                t_loss = criterion(tg_pred, shuffled_labels)
                t_loss.backward()
                optimizer.step()
                epoch_losses.append(t_loss.item())
                # update the model
                self.model.update_hiddens_all_with(x_embeddings)

            avg_epoch_loss = np.mean(epoch_losses)
            train_avg_epoch_loss_dict[epoch] = avg_epoch_loss
            
            patience = 0
            if avg_epoch_loss < min_loss:
                    min_loss = avg_epoch_loss
                    test_epoch, test_auc, test_ap = self.tgclassification_test(epoch)
                    patience = 0
            else:
                    patience += 1
                    if epoch > args.min_epoch and patience > args.patience:  # NOTE: args.min_epoch prevents it from stopping early in most cases
                        print('INFO: Early Stopping...')
                        break
                    test_epoch, test_auc, test_ap = None, None, None
            gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

            if epoch == 1 or epoch % args.log_interval == 0:
                    logger.info('==' * 30)
                    logger.info("Epoch:{}, Time: {:.3f}, GPU: {:.1f}MiB".format(epoch, avg_epoch_loss,
                                                                                            time.time() - t_epoch_start,
                                                                                            gpu_mem_alloc))
                    logger.info(
                        "Test: Epoch:{}, Loss: {:.4f}, AUC: {}, AP: {}".format(
                        test_epoch if test_epoch is not None else "N/A",
                        avg_epoch_loss,
                        f"{test_auc:.4f}" if test_auc is not None else "N/A",
                        f"{test_ap:.4f}" if test_ap is not None else "N/A"
                    ))
                    epochMessage = f"Epoch {epoch:02d} | Train Loss: {avg_epoch_loss:.4f} | Train AUCROC " + f"{test_auc:.4f}" if test_auc is not None else "N/A"
                    with open(rf"{file_visualization_path}/{args.dataset}/{args.model}/multiheadMLP_performance_{args.seed}.txt", "a") as f:
                        f.write(epochMessage + "\n")
            if isnan(t_loss):
                    print('ATTENTION: nan loss')
                    break
            
        logger.info('>> Total time : %6.2f' % (time.time() - t_total_start))
        logger.info(">> Parameters: lr:%.4f |Dim:%d |Window:%d |" % (args.lr, args.nhid, args.nb_window))

        # ------------ DEBUGGING ------------
        # save the training loss values
        partial_results_path = f'../data/output/log/{args.dataset}/{args.model}/'
        loss_log_filename = f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.pkl'
        with open(loss_log_filename, 'wb') as file:
            dump(train_avg_epoch_loss_dict, file)
        
        # plotting the training losses
        train_avg_epoch_loss_dict = load(open(loss_log_filename, 'rb'))
        train_values = train_avg_epoch_loss_dict.values()
        epoch_range = range(0, epoch)
        # plt.plot(epoch_range, train_values, label='Training Loss')
        # plt.title('Training Loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.xticks(np.arange(0, epoch, 50))
        # plt.legend(loc='best')
        # plt.show()
        # plt.savefig(f'{partial_results_path}/{args.model}_{args.dataset}_{args.seed}_train_loss.png')
        # -----------------------------------
        # -----------------------------------

        # Final Test
        # test_epoch, test_auc, test_ap = self.tgclassification_test(epoch, self.readout_scheme)
        # logger.info("Final Test: Epoch:{} , AUC: {:.4f}, AP: {:.4f}".format(test_epoch, test_auc, test_ap))


if __name__ == '__main__':
    from GraphGeneration.models.temporal_gnn.script.config import args
    from GraphGeneration.models.temporal_gnn.script.utils.util import set_random, logger, init_logger, disease_path
    from GraphGeneration.models.temporal_gnn.script.models.load_model import load_model
    from GraphGeneration.models.temporal_gnn.script.utils.data_util import loader, prepare_dir
    from GraphGeneration.models.temporal_gnn.script.inits import prepare
    
    file_visualization_path = "./GraphGeneration/scripts/Visualize"
    if not os.path.exists(f"{file_visualization_path}/{args.dataset}/{args.model}"):
        os.makedirs(rf"{file_visualization_path}/{args.dataset}/{args.model}")
    with open(f"{file_visualization_path}/{args.dataset}/{args.model}/multiheadMLP_performance_{args.seed}.txt", "w") as f:
        f.write("")
    print(f"{file_visualization_path}/{args.dataset}/{args.model}/multiheadMLP_performance_{args.seed}.txt")
    for i in range(20, args.num_snapshots):
        print("INFO: >>> Temporal Graph Classification <<<")
        print("INFO: Predict snapshot: ", i)
        print("INFO: Args: ", args)
        print("======================================")
        print("INFO: Dataset: {}".format(args.dataset))
        print("INFO: Model: {}".format(args.model))
        data = loader(dataset=args.dataset, neg_sample=args.neg_sample, targetsnapshot=i)
        args.num_nodes = data['num_nodes']
        print("INFO: Number of nodes:", args.num_nodes)
        set_random(args.seed)
        init_logger(prepare_dir(args.output_folder) + args.model + '_' + args.dataset + '_seed_' + str(args.seed) + '_log.txt')
        runner = Runner()
        runner.run()


# ----------------------
# commands to run:
# python scripts/train_tgc_end_to_end.py --model=HTGN --seed=710  --dataset=dgd --max_epoch=200