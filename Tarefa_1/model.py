
from torchinfo import summary
import torch.nn as nn
import torch

'''
Código responsável por definir a arquitetura das redes neurais
'''

class ModelFullyconnected(nn.Module):
    '''
    ModelFullyconnected é um modelo MLP (Multi-Layer Perceptron). Não usa convolução, apenas liga todos os pixeis a todos os outputs.
    '''
    def __init__(self):
        super(ModelFullyconnected, self).__init__()  #Ativa o pytorch. Carrega todas as ferramentas básicas de IA (gravar pesos, usar GPU, etc.)

        #Passamos a imagem com 28x28 pixeis (tamanho das imagens usadas) para uma linha com números (784)
        nrows = 28 
        ncols = 28
        ninputs = nrows * ncols 

        noutputs = 10 #Número de saídas possíveis

        # Criamos uma camada totalmente conectada. Ligas os 784 pixeis de entrada a cada um dos 10 resultados possíveis.
        self.fc = nn.Linear(ninputs, noutputs)

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def forward(self, x):
        # "x" corresponde á imagem que está nor formato [Batch, 1, 28, 28]

        x = x.view(x.size(0), -1) #Pega no quadrado 28x28 e coloca tudo numa linha de 784 números

        y = self.fc(x) #Damos os dados da imagem á camada totalmente conectada

        return y

    def getNumberOfParameters(self):
        #Soma o total dos "pesos". Serve apenas para ter noção da complexidade do modelo
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ModelConvNet(nn.Module):

    '''
    Ao contrário do modelo anterior (que achatava a imagem logo no início), 
    este modelo "olha" para a imagem como uma matriz (2D). 
    Ele usa "janelas" (filtros) que deslizam sobre a imagem para detetar padrões (linhas, curvas, círculos).
    '''

    def __init__(self):

        super(ModelConvNet, self).__init__()  

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        #Camada de detetores báseicos, em que aplicamos 32 filtros diferentes 
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        #Pega na informação do con1 e comprime. Passa de uma imagem 28x28 para 14x14 mas mantem as carcateisticas mais importantes
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    

        #Pega nos resultados anteriores e aplica mais 32 filtros.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        #Apõs as convoluções criamos uam camada totalmente conectada para dar a resposta
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)


        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):

        #Segue a mesma lógica do forward anterior, no entanto neste mantemos o 2D nas convoluções até ao linear que temos que passar á linha de números.

        print('Forward method called ...')

        print('Input x.shape = ' + str(x.shape))

        x = self.conv1(x)
        print('After conv1 x.shape = ' + str(x.shape))

        x = self.pool1(x)
        print('After pool1 x.shape = ' + str(x.shape))

        x = self.conv2(x)
        print('After conv2 x.shape = ' + str(x.shape))

        x = self.pool2(x)
        print('After pool2 x.shape = ' + str(x.shape))

        x = x.view(-1, 64*7*7)
        print('After flattening x.shape = ' + str(x.shape))

        x = self.fc1(x)
        print('After fc1 x.shape = ' + str(x.shape))

        y = self.fc2(x)
        print('Output y.shape = ' + str(y.shape))

        return y


class ModelConvNet3(nn.Module):

    '''
    Segue extamente a mesma lógica que o modelconvnet, mas tem uma camada extra,
    ou seja, uma envolução da anterior.
    '''

    def __init__(self):

        super(ModelConvNet3, self).__init__() 

        nrows = 28
        ncols = 28
        ninputs = nrows * ncols
        noutputs = 10

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

   
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=2)
  


        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
 


        self.fc1 = nn.Linear(128 * 2 * 2, 128)



        self.fc2 = nn.Linear(128, 10)
  

        print('Model architecture initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')
        summary(self, input_size=(1, 1, 28, 28))

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):


        x = self.conv1(x)

        x = self.pool1(x)

        x = self.conv2(x)

        x = self.pool2(x)

        x = self.conv3(x)

        x = self.pool3(x)

        x = x.view(-1, 128*2*2)


        x = self.fc1(x)

        y = self.fc2(x)

        return y

class ModelBetterCNN(nn.Module):

    '''
    ModelBetterCNN é uma  arquitetura mais robusta e otimizada em relação aos restantes modelos.
    
    Diferenças principais para os modelos anteriores:
    1. Batch Normalization (bn1, bn2, bn3): Adicionado após cada convolução. 
       Normaliza os dados para estabilizar o treino, permitindo que a rede aprenda 
       mais rápido e não fique 'presa' em mínimos locais.
       
    2. Dropout (0.5): Desliga aleatoriamente 50% dos neurónios durante o treino para forçar 
       a rede a criar caminhos redundantes e evitar o Overfitting (decorar dados).
       
    3. Maior Capacidade: A camada linear intermédia (fc1) foi aumentada para 256 neurónios 
       (vs 128 nos anteriores), permitindo processar mais informação complexa.
    '''

    def __init__(self):
        super(ModelBetterCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normaliza os 32 canais
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 

  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128) 
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) 

        self.fc1 = nn.Linear(128 * 3 * 3, 256) # Camada densa maior
        self.dropout = nn.Dropout(0.5)         # Desliga 50% dos neurónios aleatoriamente
        self.fc2 = nn.Linear(256, 10)          # 10 saídas (0 a 9)

        print('ModelBetterCNN initialized with ' + str(self.getNumberOfParameters()) + ' parameters.')

    def getNumberOfParameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        
        # Bloco 1
        x = self.conv1(x)
        x = self.bn1(x)     
        x = torch.relu(x)   # OBRIGATÓRIO: A ativação não-linear
        x = self.pool1(x)
        
        # Bloco 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # Bloco 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)

   
        x = x.view(-1, 128 * 3 * 3)
        
        # Classificação
        x = self.fc1(x)
        x = torch.relu(x)   
        x = self.dropout(x) 
        y = self.fc2(x)
        
        return y