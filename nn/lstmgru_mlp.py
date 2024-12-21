import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from utils.utils import Utils
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, average_precision_score



class LSTMGRU_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_1=64, dropout=0, hidden_dim_2=32, mlp_dim=32, num_layers_LSTM=1, num_layers_GRU=1):
        super(LSTMGRU_MLP, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.mlp_dim = mlp_dim
        self.output_dim = output_dim
        
         # Define LSTM layers
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim_1, num_layers=num_layers_LSTM, dropout=dropout)

        # Define GRU layers
        self.gru1 = nn.GRU(input_size=self.hidden_dim_1, hidden_size=self.hidden_dim_2, num_layers=num_layers_GRU, dropout=dropout)

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim_2, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, output_dim),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Ensure the input tensor is on the same device as the model
        x = x.to(next(self.parameters()).device)
        
        # Forward pass through the LSTM layers
        x, _ = self.lstm1(x)

        # Forward pass through the GRU layers
        x, _ = self.gru1(x)

        # Pass through the fully connected layers
        if self.output_dim != 1:
            x = x[:, -1, :]  # Take the output from the last time step

        x = self.mlp(x)  # Go through the MLP

        # In case we are doing binary classification, do a sigmoid activation
        if self.output_dim == 1:
            x = self.sigmoid(x.squeeze())
        
        return x
    
    
    def train_model_binary(self, model, train_loader, optimizer, criterion):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        model.train()
        epoch_loss = 0
        predictions = []
        labels = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            y = y.squeeze().float()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            predictions.append(output.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())
            
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)
        
        # Compute metrics
        train_aucroc = roc_auc_score(labels, predictions)
        train_aucpr = average_precision_score(labels, predictions)
        train_pred_labels = [1 if prob >= 0.5 else 0 for prob in predictions]  # Since accuracy needs exact labels
        train_accuracy = accuracy_score(labels, train_pred_labels)
        
        return (epoch_loss / len(train_loader)), train_aucroc, train_aucpr, train_accuracy
    
    
    def test_model_binary(self, model, test_loader, criterion, y_test, display_confusion=False):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        model.eval()
        test_loss = 0
        test_preds = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)  # Maintain hidden state across time steps
                test_preds.append(output.detach().numpy())
                y = y.squeeze().float()
                loss = criterion(output, y)
                test_loss += loss.item()

        test_preds = np.concatenate(test_preds, axis=0)  # Ensure val_preds is a flat array  # Flatten if it's a list
        test_preds = np.array(test_preds)
        test_loss /= len(test_loader)
        
        # Compute metrics
        test_aucroc = roc_auc_score(y_test, test_preds)
        test_aucpr = average_precision_score(y_test, test_preds)
        test_pred_labels = [1 if prob >= 0.5 else 0 for prob in test_preds]
        test_accuracy = accuracy_score(y_test, test_pred_labels)

        if display_confusion:
            cm = confusion_matrix(y_test, test_pred_labels)

            # Display confusion matrix
            print("Confusion Matrix:")
            # Optionally, plot the confusion matrix using seaborn for better visualization
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Shrink', 'Growth'], yticklabels=['Shrink', 'Growth'])
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            plt.title('Confusion Matrix')
            plt.show()
            plt.clf()

        return test_loss, test_aucroc, test_aucpr, test_accuracy
    
    
    def test_model_regression(self, best_model, test_loader, criterion, split_index):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        test_loss = 0
        cosine_similarities = []
        norms = []
        best_model.eval()
        time_index = split_index  # Start time index at the beginning of the test set
        predicted_embeddings = []
        predicted_embeddings_linfit = []
        real_embeddings = []
        my_utils = Utils()

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                output = best_model(x)  # Maintain hidden state across time steps
                y = y.float()
                loss = criterion(output, y)
                test_loss += loss.item()
                
                # Print time index, predicted embedding, and real embedding
                for i in range(len(x)):
                    predicted_embedding = output[i].cpu().numpy()
                    real_embedding = y[i].cpu().numpy()
                    predicted_embedding_linfit = my_utils.linear_fit(predicted_embedding)

                    predicted_embeddings_linfit.append(predicted_embedding_linfit)  # Fit a LinearRegression model for monotonically increasing behavior
                    real_embeddings.append(real_embedding)
                    predicted_embeddings.append(predicted_embedding)  # add to list for reconstruction
                    
                    # Visualize 20-dim embeddings
                    # predicted_linfit_str = '\t'.join(map(str, predicted_embedding_linfit))
                    # predicted_str = '\t'.join(map(str, predicted_embedding))
                    # real_str = '\t'.join(map(str, real_embedding))
                    # print(f"Time Index:\t{time_index}\nPredicted Embedding:\t{predicted_str}\nLinear Fit Embedding:\t{predicted_linfit_str}\nReal Embedding:\t{real_str}")
                    # print("-" * 50)
                    
                    cosine_similarities.append(my_utils.compute_cosine_similarity(predicted_embedding, real_embedding))
                    norms.append(my_utils.compute_distances(predicted_embedding, real_embedding))

                    time_index += 1
            
        # my_visualizer = Visualizer(dataset="Cosine", task="Regression")
        # for i in range(3):
        #     my_visualizer.display_embeddings_once(predicted_embeddings[i], real_embeddings[i], predicted_embeddings_linfit[i])
        
        test_loss /= len(test_loader)
        avg_norm = np.mean(norms)
        avg_cosine_similarity = np.mean(cosine_similarities)
        
        return test_loss, avg_norm, avg_cosine_similarity