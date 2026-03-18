import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=33, num_classes=2, e_dim=480, ff_dim=480, fc_dim=96, num_heads=20, num_layers=8, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(e_dim, 
                                       num_heads, 
                                       dim_feedforward=ff_dim, 
                                       dropout=dropout,
                                       activation="gelu",
                                       batch_first=True),
            num_layers
        )
        self.fc1 = nn.Linear(e_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)
    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc1(x)
        x = torch.sigmoid(self.fc2(x))
        return x

def load(model_path):
    model = TransformerClassifier()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    if torch.cuda.is_available():
        model = model.cuda()
    return model
