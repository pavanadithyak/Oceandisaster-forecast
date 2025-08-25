import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class BayesianLSTM(nn.Module):
    """
    Bayesian LSTM with Monte Carlo Dropout for uncertainty quantification
    """
    
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout_rate=0.2):
        super(BayesianLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Dropout layer for Monte Carlo sampling
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output layer
        self.linear = nn.Linear(hidden_size, 1)
        
        # Scaler for data normalization
        self.scaler = MinMaxScaler()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, x):
        """Forward pass through the network"""
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output
        lstm_out = lstm_out[:, -1, :]
        
        # Apply dropout (active during training and MC sampling)
        lstm_out = self.dropout(lstm_out)
        
        # Linear layer
        output = self.linear(lstm_out)
        
        return output
    
    def prepare_data(self, sequences, targets, batch_size=32):
        """Prepare data for training"""
        # Convert to tensors
        X = torch.FloatTensor(sequences).to(self.device)
        y = torch.FloatTensor(targets).to(self.device)
        
        # Reshape X to (batch_size, sequence_length, input_size)
        if len(X.shape) == 2:
            X = X.unsqueeze(-1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return dataloader
    
    def train_model(self, train_loader, epochs=100, learning_rate=0.001, progress_callback=None):
        """Train the Bayesian LSTM model"""
        # Set to training mode
        self.train()
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training history
        train_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in train_loader:
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(batch_X)
                
                # Calculate loss
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                # Update weights
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Average loss for this epoch
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)
            
            # Update learning rate
            scheduler.step(avg_loss)
            
            # Progress callback
            if progress_callback:
                progress_callback((epoch + 1) / epochs)
            
            # Early stopping check
            if len(train_losses) > 20:
                recent_losses = train_losses[-10:]
                if max(recent_losses) - min(recent_losses) < 1e-6:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return train_losses
    
    def predict_with_uncertainty(self, input_sequence, forecast_horizon, mc_samples=100):
        """
        Generate predictions with uncertainty quantification using Monte Carlo Dropout
        
        Args:
            input_sequence: Input sequence for prediction
            forecast_horizon: Number of steps to forecast
            mc_samples: Number of Monte Carlo samples
            
        Returns:
            predictions: Mean predictions
            uncertainties: Standard deviations (uncertainties)
        """
        # Set to training mode to keep dropout active
        self.train()
        
        # Convert input to tensor
        if isinstance(input_sequence, np.ndarray):
            input_tensor = torch.FloatTensor(input_sequence).to(self.device)
        else:
            input_tensor = input_sequence
        
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(-1)
        
        # Store predictions from multiple MC samples
        mc_predictions = []
        
        with torch.no_grad():
            for _ in range(mc_samples):
                predictions = []
                current_sequence = input_tensor.clone()
                
                # Generate multi-step predictions
                for step in range(forecast_horizon):
                    # Predict next step
                    pred = self(current_sequence)
                    predictions.append(pred.cpu().numpy().flatten()[0])
                    
                    # Update sequence for next prediction
                    # Remove first element and append prediction
                    new_input = torch.cat([
                        current_sequence[:, 1:, :],
                        pred.unsqueeze(1).unsqueeze(-1)
                    ], dim=1)
                    current_sequence = new_input
                
                mc_predictions.append(predictions)
        
        # Convert to numpy array
        mc_predictions = np.array(mc_predictions)
        
        # Calculate mean and standard deviation
        mean_predictions = np.mean(mc_predictions, axis=0)
        std_predictions = np.std(mc_predictions, axis=0)
        
        return mean_predictions, std_predictions
    
    def evaluate(self, test_loader):
        """Evaluate the model on test data"""
        self.eval()
        
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = self(batch_X)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(batch_y.cpu().numpy().flatten())
        
        return np.array(predictions), np.array(actuals)
    
    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'scaler': self.scaler
        }, filepath)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout_rate=checkpoint['dropout_rate']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.scaler = checkpoint['scaler']
        
        return model
    
    def get_model_info(self):
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout_rate': self.dropout_rate,
            'device': str(self.device)
        }
