import os
import pickle
import torch
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import EncoderCNN, DecoderRNN, EncoderGRU, DecoderGRU, EncoderATT, DecoderLSTMWithAttention, Vocabulary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set parameters
embed_size = 256
hidden_size = 512
vocab_file = "vocab.pkl"

# Load vocabulary
with open(vocab_file, "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab)

# Initialize models
encoder1 = EncoderGRU(embed_size).to(device)
decoder1 = DecoderGRU(embed_size, hidden_size, vocab_size).to(device)

encoder2 = EncoderCNN(embed_size).to(device)
decoder2 = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

encoder3 = EncoderATT(embed_size).to(device)
decoder3 = DecoderLSTMWithAttention(embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.5).to(device)

# Load trained weights
encoder1.load_state_dict(torch.load(os.path.join("./models", "encoder_gru-5.pkl")))
decoder1.load_state_dict(torch.load(os.path.join("./models", "decoder_gru-5.pkl")))

encoder2.load_state_dict(torch.load(os.path.join("./models", "encoder-5.pkl")))
decoder2.load_state_dict(torch.load(os.path.join("./models", "decoder-5.pkl")))

encoder3.load_state_dict(torch.load(os.path.join("./models", "encoder_cnn-5.pkl")))
decoder3.load_state_dict(torch.load(os.path.join("./models", "decoder_lstm_attn-5.pkl")))

encoder1.eval()
decoder1.eval()
encoder2.eval()
decoder2.eval()
encoder3.eval()
decoder3.eval()

# Image transformations
transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def clean_sentence(output, idx2word):
    sentence = ""
    for i in output:
        word = idx2word[i]
        if i == 0:
            continue
        if i == 1:
            break
        if i == 18:
            sentence = sentence + word
        else:
            sentence = sentence + " " + word
    return sentence

def generate_caption(image_tensor, encoder, decoder):
    with torch.no_grad():
        features = encoder(image_tensor).unsqueeze(1)
        output = decoder.sample(features)
    return clean_sentence(output, vocab.idx2word)

# Streamlit UI
st.title("Image Captioning using CNN-GRU, CNN-LSTM, and CNN-LSTM with Attention")

uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image_tensor = transform_test(image).unsqueeze(0).to(device)

    # Generate captions using all three models
    caption1 = generate_caption(image_tensor, encoder1, decoder1)
    caption2 = generate_caption(image_tensor, encoder2, decoder2)
    with torch.no_grad():
        features3 = encoder3(image_tensor).unsqueeze(1)
        output3 = decoder3.sample(features3)
        caption3 = clean_sentence(output3[0].cpu().numpy(), vocab.idx2word)

    st.write(f"**Caption (GRU)**: {caption1}")
    st.write(f"**Caption (LSTM)**: {caption2}")
    st.write(f"**Caption (LSTM with Attention)**: {caption3}")
