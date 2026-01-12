
import glob
import os
from random import randint
import shutil
import signal
from matplotlib import pyplot as plt
import numpy as np
import argparse
import torch
from dataset import Dataset
from torchvision import transforms
from model import ModelFullyconnected, ModelConvNet, ModelConvNet3, ModelBetterCNN
from trainer import Trainer
from datetime import datetime


def sigintHandler(signum, frame):
    """
    Função de Segurança: Permite parar o programa com CTRL+C
    sem queimar processos ou deixar ficheiros corrompidos.
    """
    print('SIGINT received. Exiting gracefully.')
    exit(0)


def main():

    # ------------------------------------
    # 1. SETUP DOS ARGUMENTOS (O Menu de Opções)
    # ------------------------------------
    #Parte onde damos a localização dos ficheiros e passamos certas configurações como o "batch size"

    
    parser = argparse.ArgumentParser()

    parser.add_argument('-df', '--dataset_folder', type=str,
                        default='/home/baldaia/Desktop/savi-2025-2026-trabalho2-grupo5/savi_datasets')
    parser.add_argument('-pe', '--percentage_examples', type=float, default=1,
                        help='Percentage of examples to use for training and testing')
    parser.add_argument('-ne', '--num_epochs', type=int, default=10,
                        help='Number of epochs for training')
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='Batch size for training and testing.')
    parser.add_argument('-ep', '--experiment_path', type=str,
                        default='/home/baldaia/Desktop/savi-2025-2026-trabalho2-grupo5/Tarefa_1/Experiments_MBCNN',
                        help='Path to save experiment results.')
    parser.add_argument('-rt', '--resume_training', action='store_true',
                        help='Resume training from last checkpoint if available.')

    args = vars(parser.parse_args())
    print(args)

    # ------------------------------------
    # 2. SEGURANÇA (Registar o Handler)
    # ------------------------------------

    # Processo de segurança para guardar os dados primeiro, antes de fechar o programa.
    # Usa a função sigintHandler criado acima.
    signal.signal(signal.SIGINT, sigintHandler)

    # ------------------------------------
    # 3. PREPARAR A PASTA DA EXPERIÊNCIA
    # ------------------------------------

    # Definimos o caminho para a pasta da esperiência 
    args['experiment_full_name'] = args['experiment_path']

    print('Starting experiment: ' + args['experiment_full_name'])

    # Cria a pasta ou usa a pasta caso ela já exista 
    os.makedirs(args['experiment_full_name'], exist_ok=True)

    # ------------------------------------
    # 4. CRIAR OS DATASETS
    # ------------------------------------
    train_dataset = Dataset(args, is_train=True)
    test_dataset = Dataset(args, is_train=False)

    # ------------------------------------
    # 5. ESCOLHER O MODELO
    # ------------------------------------
    #model = ModelFullyconnected()
    #model = ModelConvNet()
    #model = ModelConvNet3()
    model = ModelBetterCNN()

    # ------------------------------------
    # 6. INICIAR O TREINO
    # ------------------------------------
    trainer = Trainer(args, train_dataset, test_dataset, model)

    # Verificar se está tudo bem com o programa e se está tudo funcional para iniciar o processo.
    # Usamos a imagem 107 mas podiamos ter usado outra qualquer.
    image_tensor, label_gt_tensor = trainer.train_dataloader.dataset.__getitem__(107)  
    
    # Adiciona a dimensão do batch: (1, 28, 28) passa a ser (1, 1, 28, 28)
    image_tensor = image_tensor.unsqueeze(0)

    # Tenta fazer uma previsão
    label_pred_tensor = model.forward(image_tensor)

    # ------------------------------------
    # 7. EXECUTAR
    # ------------------------------------
    trainer.train()      # Inicia o treino

    trainer.evaluate()   # Inicia o teste 


if __name__ == '__main__':
    main()