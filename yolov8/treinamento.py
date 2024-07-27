import torch
from ultralytics import YOLO  # Assumindo que você está usando uma versão personalizada ou que Ultralytics suporta MPS.
import subprocess
import os

# Definindo o dispositivo como MPS se disponível, senão usa CPU
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print("Using device:", device)

# Carregar o modelo YOLOv8
# Certifique-se de que o modelo YOLO('yolov8n.pt') esteja no diretório correto ou especificado corretamente
#model = YOLO("yolov8n.pt") # Carregar um modelo pré-treinado é recomendado para treinamento
model = YOLO("runs/detect/train6/weights/best.pt")
model.to(device)           # Envia o modelo para o dispositivo especificado (MPS ou CPU)

# Configuração dos parâmetros de treinamento
data_path  = "datasets/Granito-1/granito.yaml" # Caminho para o arquivo YAML que contém as configurações do dataset
epochs     = 60                                # Número de épocas de treinamento
img_size   = 640                               # Tamanho das imagens de entrada

# Iniciar o TensorBoard apontando para a pasta runs
log_dir    = "runs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Inicia o TensorBoard
subprocess.Popen(["tensorboard", "--logdir", log_dir])

# Executando o treinamento
results        = model.train(
    data       = data_path,                   # Caminho para o arquivo YAML que contém as configurações do dataset
    epochs     = epochs,                      # Número de épocas de treinamento
    imgsz      = img_size,                    # Tamanho das imagens de entrada
    batch      = 16,                          # Tamanho do lote de treinamento
    workers    = 8,                           # Aumenta o número de workers para 8
    optimizer  = 'auto',                      # Escolhe automaticamente o melhor otimizador para o treinamento baseado no hardware e configuração.
    verbose    = True,                        # Ativa o modo verboso, que irá imprimir informações detalhadas durante o treinamento.
    device     = device.type                  # Passando o tipo de dispositivo como uma string
)

# Salvar o modelo treinado, se necessário
model.save("datasets/Granito-1/model.pt")

# Imprimir resultados de treinamento
print("Training completed. Results:", results)
