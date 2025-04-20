import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Assuming model definitions are in model.py
from model import Encoder, Decoder, IntentClassifier, Chatbot

# Load preprocessed data
with open("data/preprocessed_data.pkl", "rb") as f:
    data = pickle.load(f)
patient_padded = torch.tensor(data["patient_padded"].tolist(), dtype=torch.long)
doctor_padded = torch.tensor(data["doctor_padded"].tolist(), dtype=torch.long)
intent = torch.tensor(data["intent"].tolist(), dtype=torch.long)
dataset = TensorDataset(patient_padded, doctor_padded, intent)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model setup
VOCAB_SIZE = 10002  # Adjusted for <PAD> and <UNK>
EMBED_SIZE = 64
HIDDEN_SIZE = 128
NUM_CLASSES = 3
encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
classifier = IntentClassifier(HIDDEN_SIZE, 64, NUM_CLASSES)
model = Chatbot(encoder, decoder, classifier)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop with accuracy and plotting
num_epochs = 20
losses = []
accuracies = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (patient, doctor, target) in enumerate(dataloader):
        patient, doctor, target = patient.to(device), doctor.to(device), target.to(device)
        optimizer.zero_grad()
        hidden, cell = model.encoder(patient)
        intent_logits = model.classifier(hidden[-1])
        loss = criterion(intent_logits, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(intent_logits.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    epoch_loss = total_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    losses.append(epoch_loss)
    accuracies.append(epoch_acc)
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# Save model
torch.save(model.state_dict(), "data/model.pth")
print("Training done! Model saved to data/model.pth")

# Plot training metrics
plt.figure(figsize=(8, 6))
plt.plot(losses, label='Loss')
plt.plot([acc / 100 for acc in accuracies], label='Accuracy')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig('training_metrics.png')