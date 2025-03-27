import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Define Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, features, hidden):
        # features: (batch, num_features, hidden_size)
        # hidden: (batch, hidden_size)
        hidden = hidden.unsqueeze(1)
        energy = torch.tanh(self.attn(features) + hidden)
        attention_weights = torch.softmax(self.v(energy), dim=1)
        context = (attention_weights * features).sum(dim=1)
        return context, attention_weights

# Define LSTM Decoder with Attention
class DecoderLSTMWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderLSTMWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        h, c = torch.zeros(1, captions.size(0), features.size(-1)).to(features.device), \
               torch.zeros(1, captions.size(0), features.size(-1)).to(features.device)
        outputs = []
        for t in range(embeddings.size(1)):
            context, _ = self.attention(features, h.squeeze(0))
            lstm_input = torch.cat((embeddings[:, t, :], context), dim=1).unsqueeze(1)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            outputs.append(self.fc(lstm_out.squeeze(1)))
        return torch.stack(outputs, dim=1)

# Training loop
num_epochs = 1  # Change as needed
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderLSTMWithAttention(embed_size, hidden_size, vocab_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.embed.parameters()), lr=0.001)

for epoch in range(1, num_epochs + 1):
    decoder.train()
    encoder.train()
    progress_bar = tqdm(train_data_loader, desc=f"Epoch {epoch}", unit="batch")
    for images, captions in progress_bar:
        images, captions = images.to(device), captions.to(device)
        encoder.zero_grad()
        decoder.zero_grad()
        features = encoder(images).unsqueeze(1)
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Perplexity": f"{np.exp(loss.item()):.4f}"})
    
    print(f"Epoch {epoch} completed!")
    
