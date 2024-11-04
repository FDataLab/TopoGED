# Two Layers
class LSTMGRUPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64 , num_layers_LSTM=1, num_layers_GRU=1):
        super(LSTMGRUPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_low_scaled = (self.hidden_dim//2)
        self.num_layers = num_layers
         # Define LSTM layers
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=num_layers_LSTM)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_LSTM)

        # Define GRU layers
        self.gru1 = nn.GRU(input_size=self.hidden_dim_low_scaled, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_GRU)
        self.gru2 = nn.GRU(input_size=self.hidden_dim_low_scaled, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_GRU)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_dim_low_scaled, 100)
        self.fc2 = nn.Linear(100, output_dim)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        # Forward pass through the LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Forward pass through the GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        # Pass through the fully connected layers
        x = x[:, -1, :]  # Take the output from the last time step
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x


# train 2
def train_model(model, train_loader, optimizer, num_layers, hidden_dim, criterion):
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()

        # Reset hidden state for each batch (match batch size of x)
        hidden = (torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device),
                  torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device))

        output = model(x)


        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


# train 3
def train_model(model, train_loader, optimizer, num_layers, hidden_dim_1, hidden_dim_2, criterion):
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()

        # Reset hidden state for each batch (match batch size of x)
        hidden = ((torch.zeros(num_layers, x.size(0), hidden_dim_1).to(x.device), 
                    torch.zeros(num_layers, x.size(0), hidden_dim_1).to(x.device)),  
                    (torch.zeros(num_layers, x.size(0), hidden_dim_2).to(x.device),  
                    torch.zeros(num_layers, x.size(0), hidden_dim_2).to(x.device)),  
                    (torch.zeros(num_layers, x.size(0), hidden_dim_2).to(x.device), 
                    torch.zeros(num_layers, x.size(0), hidden_dim_2).to(x.device))
        )

        output = model(x)


        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)


# Three Layers
class LSTMGRUPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_1=64, hidden_dim_2=32, num_layers_LSTM=1, num_layers_GRU=1):
        super(LSTMGRUPredictor, self).__init__()
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2
        self.hidden_dim_low_scaled = (self.hidden_dim_2//2)
        self.num_layers = num_layers
        
         # Define LSTM layers
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim_1, num_layers=num_layers_LSTM)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim_1, hidden_size=self.hidden_dim_2, num_layers=num_layers_LSTM)
        self.lstm3 = nn.LSTM(input_size=self.hidden_dim_2, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_LSTM)

        # Define GRU layers
        self.gru1 = nn.GRU(input_size=self.hidden_dim_low_scaled, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_GRU)
        self.gru2 = nn.GRU(input_size=self.hidden_dim_low_scaled, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_GRU)
        self.gru3 = nn.GRU(input_size=self.hidden_dim_low_scaled, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_GRU)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_dim_low_scaled, 100)
        self.fc2 = nn.Linear(100, output_dim)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        # Forward pass through the LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        # Forward pass through the GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)

        # Pass through the fully connected layers
        x = x[:, -1, :]  # Take the output from the last time step
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
    
# Attention 2
class LSTMGRUPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers_LSTM=1, num_layers_GRU=1):
        super(LSTMGRUPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_low_scaled = (self.hidden_dim // 2)

        # Define LSTM layers
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=num_layers_LSTM, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_LSTM, batch_first=True)

        # Define GRU layers
        self.gru1 = nn.GRU(input_size=self.hidden_dim_low_scaled, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_GRU, batch_first=True)
        self.gru2 = nn.GRU(input_size=self.hidden_dim_low_scaled, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_GRU, batch_first=True)

        # Attention layer
        self.attention = Attention(self.hidden_dim_low_scaled)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_dim_low_scaled, 100)
        self.fc2 = nn.Linear(100, output_dim)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through the LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        # Forward pass through the GRU layers
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        # Apply attention
        context, _ = self.attention(x)

        # Use the context vector for the final prediction
        x = context[:, -1, :]  # Take the output from the last time step
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
    
# Dropout 2    
class LSTMGRUPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, dropout_prob=0.5, num_layers_LSTM=1, num_layers_GRU=1):
        super(LSTMGRUPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_dim_low_scaled = (self.hidden_dim//2)
        self.num_layers = num_layers
         # Define LSTM layers
        self.lstm1 = nn.LSTM(input_size=input_dim, hidden_size=self.hidden_dim, num_layers=num_layers_LSTM)
        self.lstm2 = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_LSTM)

        # Define GRU layers
        self.gru1 = nn.GRU(input_size=self.hidden_dim_low_scaled, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_GRU)
        self.gru2 = nn.GRU(input_size=self.hidden_dim_low_scaled, hidden_size=self.hidden_dim_low_scaled, num_layers=num_layers_GRU)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_dim_low_scaled, 100)
        self.fc2 = nn.Linear(100, output_dim)

        self.dropout = nn.Dropout(p=dropout_prob)

        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        # Forward pass through the LSTM layers
        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)

        # Forward pass through the GRU layers
        x, _ = self.gru1(x)
        x = self.dropout(x)
        x, _ = self.gru2(x)
        x = self.dropout(x)

        # Pass through the fully connected layers
        x = x[:, -1, :]  # Take the output from the last time step
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
    
# Dropout 2
def train_model(model, train_loader, optimizer, num_layers, hidden_dim, criterion):
    model.train()
    epoch_loss = 0
    for x, y in train_loader:
        optimizer.zero_grad()

        # Reset hidden state for each batch (match batch size of x)
        hidden = (torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device),
                  torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device))

        output = model(x)


        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(train_loader)