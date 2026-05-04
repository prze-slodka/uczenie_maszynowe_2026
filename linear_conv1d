from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch

# 1. Pobieramy dane (na przykładzie 20 Newsgroups)
# Dla IMDB proces jest analogiczny
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))

# 2. Tekst -> Liczby (TF-IDF)
vectorizer = TfidfVectorizer(max_features=5000) # Ograniczamy do 5k cech
X = vectorizer.fit_transform(data.data).toarray()
y = data.target

# 3. Podział na train/test i zamiana na Tensory
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

import torch.nn as nn

class TextClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        
        # Pierwsza warstwa liniowa
        self.fc1 = nn.Linear(input_dim, 128)
        
        # Conv1D (wymaga wejścia: [Batch, Channels, Length])
        # Traktujemy nasze 128 cech jako 1 kanał o długości 128
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        
        # Druga warstwa liniowa (wyjściowa)
        # Po Conv1D mamy 32 kanały po 128 elementów = 32*128
        self.fc2 = nn.Linear(32 * 128, num_classes)
        
        self.relu = nn.ReLU()
        print("Model initialized.")

    def forward(self, x):
        # x: [Batch, 5000]
        x = self.relu(self.fc1(x))    # -> [Batch, 128]
        
        # Przygotowanie pod Conv1D: dodajemy wymiar kanału
        x = x.unsqueeze(1)            # -> [Batch, 1, 128]
        
        x = self.relu(self.conv1(x))  # -> [Batch, 32, 128]
        
        # Spłaszczamy do Linear
        x = x.view(x.size(0), -1)     # -> [Batch, 32 * 128]
        
        x = self.fc2(x)               # -> [Batch, num_classes]
        return x

        print("Forward pass completed.")

# Inicjalizacja (5000 cech wejściowych, 20 kategorii wyjściowych)
model = TextClassifier(5000, 20)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Prosta pętla (20 epok)
for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train[:1000]) # Mała paczka dla testu
    loss = criterion(outputs, y_train[:1000])
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
