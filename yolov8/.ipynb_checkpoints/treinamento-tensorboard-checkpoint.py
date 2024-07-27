{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6de4dab5-58c0-43d8-abed-1651475a8eeb",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/alex/Library/Python/3.9/lib/python/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020\n",
      "  warnings.warn(\n",
      "WARNING ⚠️ Ultralytics settings reset to default values. This may be due to a possible problem with your settings or a recent ultralytics package update. \n",
      "View settings with 'yolo settings' or at '/Users/alex/Library/Application Support/Ultralytics/settings.yaml'\n",
      "Update settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Using device: mps\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "New https://pypi.org/project/ultralytics/8.2.60 available 😃 Update with 'pip install -U ultralytics'\n",
      "Ultralytics YOLOv8.0.196 🚀 Python-3.9.6 torch-2.3.1 MPS (Apple M2)\n",
      "\u001b[34m\u001b[1mengine/trainer: \u001b[0mtask=detect, mode=train, model=yolov8n.pt, data=datasets/granito/granito.yaml, epochs=10, patience=50, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=mps, workers=8, project=None, name=None, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, show=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, vid_stride=1, stream_buffer=False, line_width=None, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, boxes=True, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train20\n",
      "/Users/alex/Library/Python/3.9/lib/python/site-packages/urllib3/__init__.py:35: NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with 'LibreSSL 2.8.3'. See: https://github.com/urllib3/urllib3/issues/3020\n",
      "  warnings.warn(\n",
      "Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all\n",
      "TensorBoard 2.17.0 at http://localhost:6006/ (Press CTRL+C to quit)\n",
      "Overriding model.yaml nc=80 with nc=4\n",
      "\n",
      "                   from  n    params  module                                       arguments                     \n",
      "  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 \n",
      "  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                \n",
      "  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             \n",
      "  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                \n",
      "  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             \n",
      "  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               \n",
      "  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           \n",
      "  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              \n",
      "  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           \n",
      "  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 \n",
      " 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
      " 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
      " 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 \n",
      " 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          \n",
      " 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
      " 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  \n",
      " 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                \n",
      " 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
      " 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 \n",
      " 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              \n",
      " 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           \n",
      " 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 \n",
      " 22        [15, 18, 21]  1    752092  ultralytics.nn.modules.head.Detect           [4, [64, 128, 256]]           \n",
      "Model summary: 225 layers, 3011628 parameters, 3011612 gradients, 8.2 GFLOPs\n",
      "\n",
      "Transferred 319/355 items from pretrained weights\n",
      "\u001b[34m\u001b[1mTensorBoard: \u001b[0mStart with 'tensorboard --logdir runs/detect/train20', view at http://localhost:6006/\n",
      "Freezing layer 'model.22.dfl.conv.weight'\n",
      "\u001b[34m\u001b[1mtrain: \u001b[0mScanning /Users/alex/granito/yolov8/datasets/granito/imagens/train/labels.cache... 2034 images, 15 backgrounds, 0 corrupt: 100%|██████████| 2034/2034 [00:00<?, ?it/s]\u001b[0m\n",
      "WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 13492, len(boxes) = 14812. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.\n",
      "\u001b[34m\u001b[1mval: \u001b[0mScanning /Users/alex/granito/yolov8/datasets/granito/imagens/valid/labels.cache... 254 images, 2 backgrounds, 0 corrupt: 100%|██████████| 254/254 [00:00<?, ?it/s]\u001b[0m\n",
      "WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 1770, len(boxes) = 1980. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.\n",
      "Plotting labels to runs/detect/train20/labels.jpg... \n",
      "\u001b[34m\u001b[1moptimizer:\u001b[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... \n",
      "\u001b[34m\u001b[1moptimizer:\u001b[0m AdamW(lr=0.00125, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)\n",
      "Image sizes 640 train, 640 val\n",
      "Using 0 dataloader workers\n",
      "Logging results to \u001b[1mruns/detect/train20\u001b[0m\n",
      "Starting training for 10 epochs...\n",
      "Closing dataloader mosaic\n",
      "\n",
      "      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size\n",
      "       1/10         0G      1.735       3.47      1.988         46        640:  13%|█▎        | 17/128 [00:52<04:38,  2.51s/it]"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "from ultralytics import YOLO  # Assumindo que você está usando uma versão personalizada ou que Ultralytics suporta MPS.\n",
    "import subprocess\n",
    "import os\n",
    "\n",
    "# Definindo o dispositivo como MPS se disponível, senão usa CPU\n",
    "device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')\n",
    "print(\"Using device:\", device)\n",
    "\n",
    "# Carregar o modelo YOLOv8\n",
    "# Certifique-se de que o modelo YOLO('yolov8n.pt') esteja no diretório correto ou especificado corretamente\n",
    "model = YOLO(\"yolov8n.pt\") # Carregar um modelo pré-treinado é recomendado para treinamento\n",
    "model.to(device)           # Envia o modelo para o dispositivo especificado (MPS ou CPU)\n",
    "\n",
    "# Configuração dos parâmetros de treinamento\n",
    "data_path  = \"datasets/granito/granito.yaml\"  # Caminho para o arquivo YAML que contém as configurações do dataset\n",
    "epochs     = 10                               # Número de épocas de treinamento\n",
    "img_size   = 640                              # Tamanho das imagens de entrada\n",
    "\n",
    "# Iniciar o TensorBoard apontando para a pasta runs\n",
    "log_dir = \"runs\"\n",
    "if not os.path.exists(log_dir):\n",
    "    os.makedirs(log_dir)\n",
    "\n",
    "# Inicia o TensorBoard\n",
    "subprocess.Popen([\"tensorboard\", \"--logdir\", log_dir])\n",
    "\n",
    "# Executando o treinamento\n",
    "results = model.train(\n",
    "    data       = data_path,                   # Caminho para o arquivo YAML que contém as configurações do dataset\n",
    "    epochs     = epochs,                      # Número de épocas de treinamento\n",
    "    imgsz      = img_size,                    # Tamanho das imagens de entrada\n",
    "    batch      = 16,                          # Tamanho do lote de treinamento\n",
    "    workers    = 8,                           # Aumenta o número de workers para 8\n",
    "    optimizer  = 'auto',                      # Escolhe automaticamente o melhor otimizador para o treinamento baseado no hardware e configuração.\n",
    "    verbose    = True,                        # Ativa o modo verboso, que irá imprimir informações detalhadas durante o treinamento.\n",
    "    device     = device.type                  # Passando o tipo de dispositivo como uma string\n",
    ")\n",
    "\n",
    "# Salvar o modelo treinado, se necessário\n",
    "model.save(\"datasets/granito/model.pt\")\n",
    "\n",
    "# Imprimir resultados de treinamento\n",
    "print(\"Training completed. Results:\", results)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "60e772c7-31d0-445c-8f0f-c4cc883d7135",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3fb6d281-2b09-418d-ac76-7a9aba7a4c12",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
