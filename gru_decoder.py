import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.data as data
import numpy as np
import os
import math
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu

# Required to define 
# embed_size, hidden_size, vocab_size, train_data_loader, val_data_loader, log_file, save_every, total_step, true_sentences
embed_size = 256  
hidden_size = 512  
train_data_loader = ...
vocab_size = len(train_data_loader.dataset.vocab)
val_data_loader = ...  
log_file = "training_log.txt"  
save_every = 1 
total_step = math.ceil(len(train_data_loader.dataset) / train_data_loader.batch_sampler.batch_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Build ground-truth captions dictionary from validation annotations ---
val_annotations_file = os.path.join(cocoapi_dir, "annotations/captions_val2017.json")
from pycocotools.coco import COCO
val_coco = COCO(val_annotations_file)

# Create a dictionary mapping each image_id to its list of ground truth captions.
true_sentences = {}
ann_ids = val_coco.getAnnIds()
annotations = val_coco.loadAnns(ann_ids)
for ann in annotations:
    img_id = ann["image_id"]
    if img_id not in true_sentences:
        true_sentences[img_id] = []
    true_sentences[img_id].append(ann["caption"])
    
def bleu_score(true_sentences, predicted_sentences):
    pass


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False  # In order to use pre-trained weights
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    
    def forward(self, images):
        features = self.feature_extractor(images)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return self.bn(features)
    
class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderGRU, self).__init__()
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = torch.zeros(num_layers, 1, hidden_size)

    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(dim=1), cap_embedding), dim=1)
        gru_out, _ = self.gru(embeddings)
        outputs = self.linear(gru_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        res = []
        for i in range(max_len):
            gru_out, states = self.gru(inputs, states)
            outputs = self.linear(gru_out.squeeze(dim=1))
            _, predicted_idx = outputs.max(dim=1)
            res.append(predicted_idx.item())
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx).unsqueeze(1)
        return res

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderGRU(embed_size, hidden_size, vocab_size).to(device)

# Define loss function and optimizer
criterion = (
    nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
)
params = list(decoder.parameters()) + list(encoder.embed.parameters())
optimizer = torch.optim.Adam(params, lr=0.001)

# Training loop
f = open(log_file, "a+")
num_epochs = 1  # Change as needed

for epoch in range(1, num_epochs + 1):
    decoder.train()
    encoder.train()
    print(f"\nEpoch {epoch}/{num_epochs}")
    progress_bar = tqdm(range(1, total_step + 1), desc="Training", unit="step")
    
    for i_step in progress_bar:
        indices = train_data_loader.dataset.get_train_indices()
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        train_data_loader.batch_sampler.sampler = new_sampler
        
        images, captions = next(iter(train_data_loader))
        images, captions = images.to(device), captions.to(device)
        
        decoder.zero_grad()
        encoder.zero_grad()
        
        features = encoder(images)
        outputs = decoder(features, captions)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Perplexity": f"{np.exp(loss.item()):.4f}"})
        stats = f"Epoch [{epoch}/{num_epochs}], Step [{i_step}/{total_step}], Loss: {loss.item():.4f}, Perplexity: {np.exp(loss.item()):.4f}"
        f.write(stats + "\n")
        f.flush()
    
    # Validation Phase
    decoder.eval()
    encoder.eval()
    predicted_sentences = {}
    
    print("Performing Validation...")
    with torch.no_grad():
        val_progress_bar = tqdm(val_data_loader, desc="Validation", unit="batch")
        for img_ids, images in val_progress_bar:
            images = images.to(device)
            for i, img_id in enumerate(img_ids):
                feature = encoder(images[i].unsqueeze(0)).unsqueeze(0)
                predicted_idx_list = decoder.sample(feature)
                predicted_caption = " ".join([
                    val_data_loader.dataset.vocab.idx2word[word_idx]
                    for word_idx in predicted_idx_list
                    if word_idx not in {val_data_loader.dataset.vocab.word2idx["<start>"], val_data_loader.dataset.vocab.word2idx["<end>"], val_data_loader.dataset.vocab.word2idx.get("<pad>", -1)}
                ])
                predicted_sentences[img_id] = [predicted_caption]
    
    val_bleu = bleu_score(true_sentences, predicted_sentences)
    print(f"Validation BLEU Score: {val_bleu:.4f}")
    
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join("./models", f"decoder-{epoch}.pkl"))
        torch.save(encoder.state_dict(), os.path.join("./models", f"encoder-{epoch}.pkl"))

f.close()
