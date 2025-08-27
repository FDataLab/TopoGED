# Going to have a different set of hyperparameters for each dataset's model

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import optuna
import sqlite3

# Update path for imports
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.loader import Loader
from utils.dataset import EmbeddingDataset
from torch.utils.data import DataLoader
from nn.custom_model import Decoder

from utils.utils import Utils

import wandb

combo_map = {
    "['LSTM', 'ReLU']": ['LSTM', 'ReLU'], 
    "['GRU', 'ReLU']": ['GRU', 'ReLU'], 
    "['LSTM', 'GRU', 'ReLU']": ['LSTM', 'GRU', 'ReLU'], 
    "['RNN']": ['RNN'],
    "['RNN', 'FC']": ['RNN', 'FC'],
    "['LSTM', 'FC']": ['LSTM', 'FC'],
    "['GRU', 'FC']": ['GRU', 'FC'],
    "['LSTM', 'GRU', 'FC']": ['LSTM', 'GRU', 'FC'],
}

# Constants
os.makedirs('data/output/results/ProbabilityTesting/data', exist_ok=True)
os.makedirs('data/output/cached_model/ProbabilityTesting', exist_ok=True)
csv_file_path = os.path.abspath('data/output/results/ProbabilityTesting/data/probability_testing_bayesian_individual_regression_no_norm.csv')
model_dir = os.path.abspath('data/output/cached_model/ProbabilityTesting/ProbabilityTesting')
seed = 42  # Can change

# Write the header if the file doesn't already exist
if not os.path.isfile(csv_file_path):
    pd.DataFrame(columns=['run_id', 'dataset', 'window_size', 'seed', 'hidden_size_rnn', 'learning_rate', 'dropout', 'l2_regularization', 'batch_size', 'num_layers', 'combo', 'trained_epochs', 'train_loss', 'valid_loss', 'test_loss', 'train_avg_norm', 'val_avg_norm', 'test_avg_norm', 'train_avg_cosine_similarity','val_avg_cosine_similarity', 'test_avg_cosine_similarity',]).to_csv(csv_file_path, index=False)


# Utility function specific to probabilities
def normalize_vector_by_groups(vec):
    vec = np.array(vec, dtype=np.float32)

    # Normalize indices 0 and 1
    group1 = vec[0:2]
    sum1 = np.sum(group1)
    vec[0:2] = group1 / sum1

    group2 = vec[2:6]
    sum2 = np.sum(group2)
    vec[2:6] = group2 / sum2

    return vec


def train_and_eval(dataset, window_size, num_layer, dropout, hidden_1, lr_val, l2_val, batch_size, combo, counter, seed, csv_file_path):
    # Setup
    my_loader = Loader()
    my_utils = Utils()
    my_utils.set_seeds(seed)
    output_dim = 6  # Outputting next vector
    input_dim = 6  
    patience = 25  # Early stopping patience
    num_epochs = 500  # Max epochs to train
    
    run_name = dataset
    
    # Set up probabilities
    probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities')  # Activation doesn't matter here
    #probabilities = probabilities_df.values.tolist()
    probabilities = probabilities_df.to_numpy(dtype=np.float32)

    #normalized = np.array([normalize_vector_by_groups(row) for row in probabilities])
    #probabilities = normalized

    # Probabilities to return
    all_real_embeddings = []
    all_pred_embeddings = []            
                
    run_name = run_name + '_' + str(counter)    
        
    # Split data 80/10/10
    n = len(probabilities)

    # Calculate split indices
    train_end = int(0.8 * n)  # 80% train
    val_start = train_end - window_size  # val starts after gap
    val_end = int(0.9 * n)  # 10% val
    test_start = val_end - window_size  # test starts after gap

    X_train = probabilities[:train_end]
    X_val = probabilities[val_start:val_end]
    X_test = probabilities[test_start:]

    train_dataset = EmbeddingDataset(X_train, k=window_size)
    valid_dataset = EmbeddingDataset(X_val, k=window_size)
    test_dataset = EmbeddingDataset(X_test, k=window_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                                        
    # Initialize wandb
    run = wandb.init(
        project="bayesian_testing_probabilities_no_norm", 
        name = run_name, 
        config={
        'dataset': dataset,
        'num_layers': num_layer,
        'dropout': dropout,
        'l2_regularization': l2_val,
        'hidden_size_rnn': hidden_1,
        'learning_rate': lr_val,
        'seed': seed,
        'window_size': window_size,
        'model': combo
        },
        reinit=True)
            
    no_improvement_counter = 0  # Number of epochs that we haven't seen an improvement in the validation AUCROC
    model = Decoder(in_channels=input_dim, out_channels=output_dim, hids_size_rnn=[hidden_1], hids_size_other=[output_dim], num_layers=[num_layer], layers=combo, bias=[True], dropout=[dropout])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_val, weight_decay=l2_val)
    criterion = nn.MSELoss() 
    
    curr_batch_best_loss = float('inf')    
     
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        cosine_similarities = []
        norms = []
        time_index = 0  # Start time index at the beginning of the train set
        predicted_embeddings = []
        real_embeddings = []

        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            output = output[:, -1, :]
            output = output.squeeze(1)
            
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            # Print time index, predicted embedding, and real embedding
            for i in range(len(x)):
                predicted_embedding = output[i].detach().cpu().numpy()
                real_embedding = y[i].detach().cpu().numpy()

                real_embeddings.append(real_embedding)
                predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                
                try:
                    cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                except:
                    cosine_similarities.append(float('nan'))
                try:
                    norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))
                except:
                    norms.append(float('nan'))

                all_pred_embeddings.append(predicted_embedding.tolist())
                all_real_embeddings.append(real_embedding.tolist())

                time_index += 1
        
        
        train_avg_norm = np.nanmean(norms)
        train_avg_cosine_similarity = np.nanmean(cosine_similarities)
        train_loss = (epoch_loss / len(train_loader))
                
        # Validation
        model.eval()
        valid_loss = 0
        cosine_similarities = []
        norms = []
        time_index = train_end  # Start time index at the beginning of the test set
        predicted_embeddings = []
        real_embeddings = []

        with torch.no_grad():
            for x, y in valid_loader:
                output = model(x)  # Maintain hidden state across time steps
                output = output[:, -1, :]
                output = output.squeeze(1)
                y = y.float()
                loss = criterion(output, y)
                valid_loss += loss.item()
                
                # Print time index, predicted embedding, and real embedding
                for i in range(len(x)):
                    predicted_embedding = output[i].detach().cpu().numpy()
                    real_embedding = y[i].detach().cpu().numpy()

                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                    
                    try:
                        cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                    except:
                        cosine_similarities.append(float('nan'))
                    try:
                        norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))
                    except:
                        norms.append(float('nan'))

                    all_pred_embeddings.append(predicted_embedding.tolist())
                    all_real_embeddings.append(real_embedding.tolist())

                    time_index += 1
            
        valid_loss /= len(valid_loader)
        val_avg_norm = np.nanmean(norms)
        val_avg_cosine_similarity = np.nanmean(cosine_similarities)
        
                
        # Testing        
        model.eval()
        test_loss = 0
        cosine_similarities = []
        norms = []
        time_index = val_end  # Start time index at the beginning of the test set
        predicted_embeddings = []
        real_embeddings = []

        with torch.no_grad():
            for x, y in test_loader:
                output = model(x)  # Maintain hidden state across time steps
                output = output[:, -1, :]
                output = output.squeeze(1)
                y = y.float()
                loss = criterion(output, y)
                test_loss += loss.item()
                                # Print time index, predicted embedding, and real embedding
                for i in range(len(x)):
                    predicted_embedding = output[i].detach().cpu().numpy()
                    real_embedding = y[i].detach().cpu().numpy()

                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                    
                    
                    try:
                        cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                    except:
                        cosine_similarities.append(float('nan'))
                    try:
                        norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))
                    except:
                        norms.append(float('nan'))            

                    all_pred_embeddings.append(predicted_embedding.tolist())
                    all_real_embeddings.append(real_embedding.tolist())

                    time_index += 1
            
        test_loss /= len(test_loader)
        test_avg_norm = np.nanmean(norms)
        test_avg_cosine_similarity = np.nanmean(cosine_similarities)
                
        # Stores our current epoch 'steps' results
        to_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
            'test_loss': test_loss,
            'train_avg_norm': train_avg_norm,
            'train_avg_cosine_similarity': train_avg_cosine_similarity,
            'val_avg_norm': val_avg_norm,
            'val_avg_cosine_similarity': val_avg_cosine_similarity,
            'test_avg_norm': test_avg_norm,
            'test_avg_cosine_similarity': test_avg_cosine_similarity,
        }    
                
        # Log each epoch results
        wandb.log(to_log)

        # Optimize for the best aucroc
        if valid_loss <= curr_batch_best_loss:
            curr_batch_best_loss = valid_loss
            
            # Save for dataframe
            best_moment_row = {
                'run_id': run.name,  # For checking Wandb Logs
                'dataset': dataset,
                'window_size': window_size,
                'seed': seed,
                'hidden_size_rnn': hidden_1,
                'learning_rate': lr_val,
                'dropout': dropout,
                'l2_regularization': l2_val,
                'batch_size': batch_size,
                'num_layers': num_layer,
                'model': combo,
                'trained_epochs': epoch + 1,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'test_loss': test_loss,
            }
            if(dataset == 'cosine'):
                print(f'SAVING FROM RUN {run.name}')
                columns = [f"{prefix}_{i}" for prefix in ["node", "edges", "weights"] for i in range(1, 11)]
                tmp_df = pd.DataFrame(real_embeddings, columns = columns)
                tmp_df.to_csv('data/output/results/RegressionTesting/exampleEmbeddings/cosine_real_ex.csv')
                tmp_df = pd.DataFrame(predicted_embeddings, columns = columns)
                tmp_df.to_csv('data/output/results/RegressionTesting/exampleEmbeddings/cosine_pred_ex.csv')
                
                            
        # Early stopping only after 50 epochs
        if epoch >= 100:
            if valid_loss <= curr_batch_best_loss:
                no_improvement_counter = 0
                curr_batch_best_loss = valid_loss
            else:
                no_improvement_counter += 1
                
            if no_improvement_counter == patience:
                print(f'Training ending at epoch number: {epoch + 1}')
                break
    
    # Save the best moment from this training
    pd.DataFrame([best_moment_row]).to_csv(csv_file_path, mode='a', header=False, index=False)
    
    return best_moment_row['train_loss'], best_moment_row['valid_loss'], all_pred_embeddings, all_real_embeddings


def objective(trial, dataset):
    # Suggest hyperparameters
    window_size = trial.suggest_int('window_size', 5, 30)
    dropout = trial.suggest_float('dropout', 0.01, 0.5) # 
    hidden_1 = trial.suggest_categorical('hidden_1', [32, 64, 128, 256, 512, 1024])  # Since it doesnt matter
    num_layers = trial.suggest_int('num_layers', 2, 4)
    lr_val = trial.suggest_float('lr_val', 1e-4, 1e-2, log=True) # Expand range
    l2_val = trial.suggest_float('l2_val', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    model = trial.suggest_categorical('combo', [
        "['RNN']",
        "['RNN', 'FC']",
        "['LSTM', 'FC']",
        "['GRU', 'FC']",
        "['LSTM', 'GRU', 'FC']"
    ])   
    model = combo_map[model]
        
    # # Necessary for making sure window isn't too large
    # my_loader = Loader()
    # probabilities_df = my_loader.load_data(dataset, activation='Degree', type='probabilities')  # Activation doesn't matter here
    # probabilities = probabilities_df.values.tolist()
    # data_len = len(probabilities)
        
    # if int(0.1 * len(data_len)):
    #     raise optuna.TrialPruned("Window size is too large for dataset")
        
    # Call your train_and_eval function for each dataset
    train_loss, val_loss, pred_embeddings, real_embeddings = train_and_eval(
        dataset=dataset,
        window_size = window_size,
        num_layer=num_layers,
        dropout=dropout,
        hidden_1=hidden_1,
        lr_val=lr_val,
        l2_val=l2_val,
        batch_size=batch_size,
        combo=model,
        counter=trial.number,
        seed=42,
        csv_file_path=csv_file_path,
    )
    loss_score = (train_loss * 0.4 + val_loss * 0.6)  # Play with these numbers a bit, (0.2, 0.8) and (0.4, 0.6)

    return loss_score, pred_embeddings, real_embeddings
    

def main():
    os.environ["WANDB_API_KEY"] = "6a5ccf040a6c90944032e58878e46c19d673cdb0"
    wandb.init(project="Probabilities", name="probabilities_bayesian_testing_individual_dataset_4060_no_norm")
    
    
    datasets = [ 'networkaeternity', 'networkiconomi', 'CollegeMsg', 'networkcindicator', 'networkdgd', 'networkaion', 'networkbancor', 'networkadex', 'networkcentra', 'networkcoindash', 'mathoverflow', 'Reddit_B',  'networkaragon',]    
    
    # Dictionary to store best results for each dataset
    best_results = {}

    # _4060 SIGNIFIES THAT WE ARE USING 40% TRAIN AND 60% VAL IN OUR BAYESIAN SCORING
    for dataset in datasets:
        STORAGE = f"sqlite:///./output/cached_model/ProbabilityTesting/bayesianSave/model_selection_{dataset}_4060)_no_norm.db"  # Where we save the study
        os.makedirs(os.path.dirname(f'output/cached_model/ProbabilityTesting/bayesianSave/model_selection_{dataset}_4060_no_norm.db'), exist_ok=True)
        print(f"Optimizing for dataset: {dataset}")
        
        study_name = f"probabilities_model_selection_{dataset}_4060_no_norm"  # Unique study name for each dataset
        
        # try:
        #     optuna.delete_study(study_name=study_name, storage=STORAGE)
        #     print(f"Study '{study_name}' deleted.")
        # except KeyError:
        #     print(f"Study '{study_name}' does not exist, nothing to delete.")
        
        study = optuna.create_study(
            study_name=study_name, 
            storage=STORAGE, 
            direction="minimize", 
            load_if_exists=True
        )
        
        best_value = float("inf")
        best_pred_embeddings = None
        best_real_embeddings = None

        try:
            for i in range(500):
                trial = study.ask()
                try:
                    loss, pred_embeds, real_embeds = objective(trial, dataset)
                    study.tell(trial, loss)

                    if loss < best_value:
                        best_value = loss
                        best_pred_embeddings = pred_embeds
                        best_real_embeddings = real_embeds

                except Exception as e:
                    print(f"Trial {i} failed: {e}")
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
        finally:
            # This runs even if interrupted by Ctrl+C or errors
            if best_pred_embeddings is not None and best_real_embeddings is not None:
                save_dir = f"data/input/cached/{dataset}/PredProbabilities/"
                os.makedirs(save_dir, exist_ok=True)

                # pred_df = pd.DataFrame(best_pred_embeddings)
                # real_df = pd.DataFrame(best_real_embeddings)

                # pred_df.to_csv(os.path.join(save_dir, f"{dataset}_pred_probabilities.csv"), index=False)
                # real_df.to_csv(os.path.join(save_dir, f"{dataset}_real_probabilities.csv"), index=False)

                print(f"Saved best trial embeddings to CSV for dataset {dataset} before exit.")
            else:
                print(f"No embeddings to save for dataset {dataset} before exit.")
        
        # # Pass dataset to the objective function
        # study.optimize(lambda trial: objective(trial, dataset), n_trials=500)
        
        # # Save the best trial for the current dataset
        # best_results[dataset] = study.best_trial

        # print(f"Best trial for {dataset}: {study.best_trial}")
        # print(f"Saving the best embeddings for dataset: {dataset}")
        
        # # Save predicted and real embeddings
        # pred_df = pd.DataFrame(study.best_trial.user_attrs["pred_embeddings"])
        # real_df = pd.DataFrame(study.best_trial.user_attrs["real_embeddings"])

        # save_dir = f"data/input/cached/{dataset}/PredEmbeddings/"
        # os.makedirs(save_dir, exist_ok=True)

        # pred_df.to_csv(os.path.join(save_dir, f"{dataset}_pred_embeddings.csv"), index=False)
        # real_df.to_csv(os.path.join(save_dir, f"{dataset}_real_embeddings.csv"), index=False)
        

    # Print all results
    print("\nBest results for each dataset:")
    for dataset, trial in best_results.items():
        print(f"{dataset}: {trial}")
    
    
if __name__ == "__main__":
    main()