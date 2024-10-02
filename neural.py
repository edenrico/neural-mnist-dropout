import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Normalizando os dados 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Carregadores de dados
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super(DeepNeuralNetwork, self).__init__()
        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
        # Dropout com probabilidade de desligamento de 50%
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        #apenas o ajuste
        x = x.view(-1, 28*28) #1 camada: recebe os dados e aprende diante dos dados
         
        # Camadas de ativação ReLU com Dropout
        x = F.relu(self.fc1(x))#recebe os neuronios da primeira e trasnforma em 50%, extraindo o alto nível
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))#divide por 50% aumentando a complexidade e construção contínua
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x)) #camada final de decisão (sem ativação pq estamos usando a entropy)
        x = self.dropout(x)
        
        x = self.fc4(x)
        
        return x


#Definindo a funão de perda EntropyLoss e o otimizador, 
#stanford course - Gradiente - ajuste para a realidade mais próxima
model = DeepNeuralNetwork()

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#rede neural in training
epochs = 10

for epoch in range(epochs):
    running_loss = 0
    for images, labels in train_loader:
        # Zerar gradientes
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass e otimização
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
    
#AVALIAÇÃO
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Acurácia no conjunto de teste: {100 * correct / total}%')



#conjunto de dados(dataset), definição da rede neural ( dropout escolhido ) camadas de ativações, função de perda ( entropy ), treinamento e avaliação
#passos ml  
#50% de dropout nas camadas para evitar que ela dependa muito de algumas unidades só 