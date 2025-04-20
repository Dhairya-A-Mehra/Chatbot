import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
    def forward(self, x):
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden, cell):
        x = self.embedding(x)  # Shape: (batch_size, 1, embed_size)
        output, (hidden, cell) = self.lstm(x, (hidden, cell))  # hidden, cell updated
        output = self.fc(output)  # Shape: (batch_size, 1, vocab_size)
        return output, hidden, cell

class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Chatbot(nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(src.device)
        hidden, cell = self.encoder(src)  # Shape: (1, batch_size, hidden_size)
        
        # Use hidden[-1] directly for intent, no squeeze needed
        intent_input = hidden[-1]  # Shape: (batch_size, hidden_size)
        intent_logits = self.classifier(intent_input)
        
        x = trg[:, 0].unsqueeze(1)  # Shape: (batch_size, 1)
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(x, hidden, cell)  # hidden, cell shape: (1, batch_size, hidden_size)
            outputs[:, t] = output.squeeze(1)  # Remove the sequence dimension
            x = trg[:, t].unsqueeze(1) if torch.rand(1).item() < teacher_forcing_ratio else output.argmax(2)
        
        return outputs, intent_logits