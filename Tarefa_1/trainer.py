import glob
import os
import zipfile
from matplotlib import pyplot as plt
import numpy as np
import requests
import seaborn
import torch
from colorama import init as colorama_init
from colorama import Fore, Style
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import json
from tqdm import tqdm
from sklearn import metrics 

'''
Código responsável por "treinar" a rede nueral
'''

class Trainer():

    def __init__(self, args, train_dataset, test_dataset, model):


        self.args = args
        self.model = model

        # 1. DATALOADERS
        # O DataLoader é quem entrega os "batches" de imagens ao modelo.
        # shuffle=True no treino baralha a ordem para o modelo não decorar a ordem.
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args['batch_size'],
            shuffle=True)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args['batch_size'],
            shuffle=False)
        
        # 2.FUNÇÃO DE ERRO 
        # Indica o quão errado o modelo está.
        # Estamos a usar MSE (Mean Squared Error). Basicamente calcula a distância 
        # entre a resposta do modelo e a resposta certa.ste)
        self.loss = nn.MSELoss() 

        # 3.OTIMIZADOR 
        # Ajusta os pesos.
        # lr=0.001 (Learning Rate): O tamanho do passo que damos em direção à solução.
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                          lr=0.001)

        #Opção entre querer continuar o treino ou começar um novo
        if self.args['resume_training']:
            self.loadTrain()
        else:
            self.train_epoch_losses = []
            self.test_epoch_losses = []
            self.epoch_idx = 0

    def train(self):
        """
        Executa o ciclo principal de treino e validação.
        Percorre o dataset várias vezes (Epochs), alternando entre:
        1. Treino (Ajustar pesos)
        2. Teste (Medir desempenho sem ajustar)
        """

        print('Training started. Max epochs = ' + str(self.args['num_epochs']))

        # CICLO DAS ÉPOCAS
        for i in range(self.epoch_idx, self.args['num_epochs']): 
            self.epoch_idx = i
            print('\nEpoch index = ' + str(self.epoch_idx))
            
            # ============================================
            # FASE 1: TREINO (O Estudo)
            # ============================================
            self.model.train() # Inicia o treino
            train_batch_losses = []
            num_batches = len(self.train_dataloader)
            
            #é nerta parte que vamos realizar o treino 
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(enumerate(self.train_dataloader), total=num_batches):

                # 1. Forward Pass (A Previsão)
                label_pred_tensor = self.model.forward(image_tensor)

                 
                #2. Transforma os números brutos em percentagens (0 a 1) para comparar com a Label
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)


                # 3. Cálculo do Erro
                # Quão longe a previsão está da verdade?
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                train_batch_losses.append(batch_loss.item())

                # 4. Otimização (A Aprendizagem)
                self.optimizer.zero_grad() # Limpa os gradientes antigos (reset)
                batch_loss.backward()      # Calcula a culpa de cada neurónio no erro
                self.optimizer.step()      # Atualiza os pesos na direção certa para reduzir o erro

            # ============================================
            # FASE 2: TESTE (O Exame)
            # ============================================
            self.model.eval() #Inicia o teste
            test_batch_losses = []
            num_batches = len(self.test_dataloader)

            # Nota: Aqui NÃO há optimizer.step() nem backward(). Só observamos.
            # É neste parte que vamos realizar o teste
            for batch_idx, (image_tensor, label_gt_tensor) in tqdm(enumerate(self.test_dataloader), total=num_batches):
                label_pred_tensor = self.model.forward(image_tensor)
                label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

                #Calculamos o erro para saber o resultados do teste
                batch_loss = self.loss(label_pred_probabilities_tensor, label_gt_tensor)
                test_batch_losses.append(batch_loss.item())

            # ============================================
            # FIM DA ÉPOCA (Relatórios)
            # ============================================
            print('Finished epoch ' + str(i) + ' out of ' + str(self.args['num_epochs']))
            
            # Calcula a média dos erros de todos os batches desta época
            train_epoch_loss = np.mean(train_batch_losses)
            self.train_epoch_losses.append(train_epoch_loss)

            test_epoch_loss = np.mean(test_batch_losses)
            self.test_epoch_losses.append(test_epoch_loss)

            # Atualiza o gráfico e guarda o progresso (Checkpoint)
            self.draw()
            self.saveTrain()

        print('Training completed.')

        # No final de tudo, corre as métricas avançadas (Matriz de Confusão, etc.)
        self.evaluate() 

    def loadTrain(self):
        '''
        Lê o ficheiro guardado para continuar o treino exatamente de onde parou.
        '''
        print('Resuming training from last checkpoint.')
        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        if not os.path.exists(checkpoint_file):
            raise ValueError('Checkpoint file not found: ' + checkpoint_file)
        checkpoint = torch.load(checkpoint_file, weights_only=False)
        self.epoch_idx = checkpoint['epoch_idx']
        self.train_epoch_losses = checkpoint['train_epoch_losses']
        self.test_epoch_losses = checkpoint['test_epoch_losses']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def saveTrain(self):
        '''
        Vai guardando num ficheiro .pk1 os dados do treino ao logo deste para caso haja alguma falha e o treino parar,
        depois possa continuar de onde parou. 
        Se o erro de teste for o mais baixo de sempre, guarda também uma cópia especial chamada best.pkl.
        '''
        checkpoint = {}
        checkpoint['epoch_idx'] = self.epoch_idx
        checkpoint['train_epoch_losses'] = self.train_epoch_losses
        checkpoint['test_epoch_losses'] = self.test_epoch_losses
        checkpoint['model_state_dict'] = self.model.state_dict()
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()

        checkpoint_file = os.path.join(self.args['experiment_full_name'], 'checkpoint.pkl')
        torch.save(checkpoint, checkpoint_file)

        if self.test_epoch_losses[-1] == min(self.test_epoch_losses):
            best_file = os.path.join(self.args['experiment_full_name'], 'best.pkl')
            torch.save(checkpoint, best_file)

    def draw(self):
        '''
        Gera e guarda o gráfico de evolução do treino. Desenha a curva de Erro de Treino (Vermelho) vs Erro de Teste (Azul).
        '''

        plt.figure(1)
        plt.clf()
        plt.title("Training Loss vs epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        axis = plt.gca()
        axis.set_xlim([1, self.args['num_epochs']+1])
        axis.set_ylim([0, 0.1])
        
        xs = range(1, len(self.train_epoch_losses)+1)
        plt.plot(xs, self.train_epoch_losses, 'r-', linewidth=2)
        plt.plot(xs, self.test_epoch_losses, 'b-', linewidth=2)

        best_epoch_idx = int(np.argmin(self.test_epoch_losses))
        plt.plot([best_epoch_idx, best_epoch_idx], [0, 0.5], 'g--', linewidth=1)
        plt.legend(['Train', 'Test', 'Best'], loc='upper right')
        plt.savefig(os.path.join(self.args['experiment_full_name'], 'training.png'))

    def evaluate(self):
        """
        Realiza a avaliação final do modelo ('O Exame Final').
        Diferente do teste durante o treino (que só vê a Loss), aqui calculamos:
        1. Matriz de Confusão (Quem é confundido com quem?)
        2. Precision, Recall e F1-Score (Métricas detalhadas por classe).
        """

        # -----------------------------------------
        # 1. PREPARAÇÃO E RECOLHA DE DADOS
        # -----------------------------------------
        self.model.eval() #Inicia o teste
        num_batches = len(self.test_dataloader)

        # Listas para guardar o histórico completo do dataset
        gt_classes = []        # Ground Truth: As respostas certas
        predicted_classes = [] # Predictions: O que o modelo achou

        print('Evaluating model on test set...')

        # Percorrer todo o dataset de teste sem calcular gradientes (tqdm cria a barra de progresso)
        for batch_idx, (image_tensor, label_gt_tensor) in tqdm(enumerate(self.test_dataloader), total=num_batches):
    
            # Ground Truth
            # label_gt_tensor vem em formato One-Hot (ex: [0, 0, 1, 0...])
            # .argmax(dim=1) descobre a posição do '1' e converte para número inteiro (ex: 2)
            batch_gt_classes = label_gt_tensor.argmax(dim=1).tolist()


            # Previsão do modeleo
            label_pred_tensor = self.model.forward(image_tensor)


            # Converter logits em probabilidades (0% a 100%)
            label_pred_probabilities_tensor = torch.softmax(label_pred_tensor, dim=1)

            # Descobrir qual a classe com maior probabilidade ( decisão final do modelo)
            batch_predicted_classes = label_pred_probabilities_tensor.argmax(dim=1).tolist()

            #Adicionar ás listas criadas
            gt_classes.extend(batch_gt_classes)
            predicted_classes.extend(batch_predicted_classes)

        # -----------------------------------------
        # 2. Matriz de Confusão 
        # -----------------------------------------

        #Matriz confusão- Matriz que analisa entre o real e o previsto
        confusion_matrix = metrics.confusion_matrix(gt_classes, predicted_classes)
        
        plt.figure(2)

        # Desenha o Heatmap
        # annot=True: Escreve os números dentro dos quadrados
        # fmt='d': Formato número inteiro
        class_names = [str(i) for i in range(10)]
        seaborn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(self.args['experiment_full_name'], 'confusion_matrix.png'))

        # -----------------------------------------
        # 3. RELATÓRIO ESTATÍSTICO 
        # -----------------------------------------
        # O classification_report calcula automaticamente:
        # - Precision: Quando diz que é '5', acerta quantas vezes?
        # - Recall: De todos os '5' que existem, quantos encontrou?
        # - F1-Score: A média equilibrada entre os dois.
        report = metrics.classification_report(gt_classes, predicted_classes, output_dict=True)
        
        # Mostra a tabela no terminal 
        print(metrics.classification_report(gt_classes, predicted_classes))

        # Guarda as estatísticas num ficheiro JSON
        json_filename = os.path.join(self.args['experiment_full_name'], 'statistics.json')
        with open(json_filename, 'w') as f:
            json.dump(report, f, indent=4)
        
        print('Statistics saved to ' + json_filename)