import glob
import os
import zipfile
import numpy as np
import requests
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms

'''
Código responsável por ir buscar as imagens e processar os dados.
'''

class Dataset(torch.utils.data.Dataset):

    def __init__(self, args, is_train):

        self.args = args
        self.train = is_train

        # 1. DEFINIR O CAMINHO
        # Decide se vamos buscar a pasta 'train' ou 'test'
        print(args['dataset_folder'])
        split_name = 'train' if is_train else 'test'
        image_path = os.path.join(args['dataset_folder'], split_name, 'images/')

        print('image path is: ' + image_path)

        # 2. LISTAR OS FICHEIROS (O Inventário)
        # O glob procura todos os ficheiros que acabam em .jpg naquela pasta
        self.image_filenames = glob.glob(image_path + "/*.jpg")
        self.image_filenames.sort()  # Ordena para garantir que a ordem é sempre igual

        # 3. LER AS RESPOSTAS (Labels)
        self.labels_filename = os.path.join(
            args['dataset_folder'], split_name, 'labels.txt')
        self.labels = []  
        
        # Abre o ficheiro de texto onde está a correspondência "imagem -> número"
        with open(self.labels_filename, "r") as f:  
            for line in f:
                parts = line.strip().split()   # Separa a linha por espaços
                label = float(parts[1])        # Pega na 2ª coluna (onde está o número da classe)
                self.labels.append(label)      # Guarda na lista 

        # Reduzir a quantidade de exemplos se necessário
        num_examples = round(len(self.image_filenames) * args['percentage_examples'])
        self.image_filenames = self.image_filenames[0:num_examples]
        self.labels = self.labels[0:num_examples]

        # 4. A FERRAMENTA DE CONVERSÃO
        # Prepara a ferramenta que vai transformar imagens JPG em Tensores matemáticos
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        # Devolve o número total de imagens na lista
        return len(self.image_filenames)

    def __getitem__(self, idx):
        '''
        Esta parte é responsável por receber um número(idx) e devolver a imagem e label,
        correspondentes, formatados.
        '''

        # ----------------------------
        # 1. TRATAR A LABEL 
        # ----------------------------

        # O modelo tem 10 neurónios de saída. Não podemos dar-lhe apenas o número "3".
        # Temos de criar um vetor de 10 posições onde a posição 3 está ligada (1).
        label_index = int(self.labels[idx])
        label = [0]*10          # Cria: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        label[label_index] = 1  # Fica: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

        # Converte para Tensor
        label_tensor = torch.tensor(label, dtype=torch.float)

        # ----------------------------
        # 2. TRATAR A IMAGEM
        # ----------------------------
        image_filename = self.image_filenames[idx]

        # Abre a imagem e converte para 'L' (Luminance/Grayscale).
        # Isto garante que a imagem tem apenas 1 canal de cor (Preto e Branco), 
        # tal como o nosso modelo espera (in_channels=1).
        image = Image.open(image_filename).convert('L')  

        # Transforma a imagem (Pixeis 0-255) num Tensor (Valores 0.0-1.0)
        image_tensor = self.to_tensor(image)

        return image_tensor, label_tensor