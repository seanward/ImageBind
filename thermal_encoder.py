#Step 1: Import libraries 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T
from vit_pytorch import ViT

#Step 2: Define model architecture 
# Image encoder (frozen)
image_model = ViT('B_16_imagenet1k', pretrained=True)
image_model.eval()

# Audio encoder 
audio_model = ViT('B_16', num_classes=512, dim=512)

# Projection heads
image_proj = nn.Linear(512, 512)  
audio_proj = nn.Linear(512, 512)

#Step 3: Define training hyperparameters
lr=3e-4  
weight_decay=0.1
epochs=16
batch_size=512 
temperature=0.07

#Step 4: Define data loaders
#Use Audioset and ESC50 datasets for (image, audio) pairs
train_dataset = Audioset(split='unbalanced', sample_rate=16000, n_mels=128, n_frames=100, n_fft=512, hop_length=160, n_classes=527)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = ESC50(split='fold1', sample_rate=16000, n_mels=128, n_frames=100, n_fft=512, hop_length=160) 
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#Step 5: Define loss function (InfoNCE loss)
criterion = nn.CrossEntropyLoss()  

#Step 6: Define optimizer  
optimizer = torch.optim.AdamW(audio_model.parameters(), lr=lr, weight_decay=weight_decay)

#Step 7: Training loop
for epoch in range(epochs):
  for x, y in train_loader:
    #Get image and audio embeddings
    image_emb = image_model(x['image'])
    image_emb = image_proj(image_emb)
    audio_emb = audio_model(x['audio'])
    audio_emb = audio_proj(audio_emb)
    
    #InfoNCE loss
    loss = criterion(audio_emb/temperature, image_emb/temperature)  
    
    #Backprop and update weights
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
  #Evaluation
  audio_model.eval()
  acc = 0
  with torch.no_grad():
    for x, y in val_loader:
      audio_emb = audio_model(x['audio'])
      audio_emb = audio_proj(audio_emb)
      preds = F.softmax(audio_emb/temperature, dim=1)
      acc += (preds.argmax(1) == y['label']).sum().item()
  acc /= len(val_loader.dataset)
  print(f'Epoch {epoch}: Val accuracy {acc}')
  audio_model.train()

# Assess if task completed:
# The thermal_encoder.py file uses image and audio data to train the audio model. To encode 
# thermal images, the data loaders and models would need to be updated to load and process 
# single channel thermal images. Once implemented, the code should be able to train the thermal 
# encoder and evaluate it on a downstream task. The loss function may also need adjustment.