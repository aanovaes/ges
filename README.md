## Projeto de Detecção de Defeitos em Chapas de Granito usando YOLOv8

### Descrição

Este projeto utiliza o modelo YOLOv8 para detectar defeitos em chapas de granito. O objetivo é identificar e classificar defeitos como veios e furos, calcular uma pontuação para a chapa e salvar os resultados em formato JSON, além de gerar imagens com as detecções destacadas.

### Estrutura do Projeto

datasets/ - Diretório contendo os conjuntos de dados de imagens de chapas de granito.
results/ - Diretório onde os resultados das detecções e as imagens processadas serão salvos.
scripts/ - Diretório contendo os scripts para treinamento, inferência e processamento de imagens.
configs/ - Diretório contendo arquivos YAML com as configurações dos modelos.

### Requisitos

Python 3.8+
Bibliotecas Python:
ultralytics
matplotlib
opencv-python
pyyaml
tabulate
Instalação

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio
```
2. Crie e ative um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
```
3.Instale as dependências:
```bash
pip install -r requirements.txt
```

### Configuração

1. Coloque suas imagens no diretório datasets/.
2. Ajuste o arquivo de configuração YAML (configs/yolov8_config.yaml) conforme necessário.

## Uso

### Treinamento
Para treinar o modelo, use o script de treinamento:

```bash
python scripts/train.py --config configs/yolov8_config.yaml
```
### Inferência
Para rodar a inferência em uma imagem e gerar os resultados:

```bash
python scripts/infer.py --config configs/yolov8_config.yaml --image-path path/to/your/image.jpg
```
### Exemplo de Script de Inferência
Aqui está um exemplo de script para rodar a inferência e gerar as saídas desejadas:

```python
import json
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import os
from tabulate import tabulate

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

def generate_quadrant_names(rows, cols):
    quadrant_names = []
    for r in range(rows):
        for c in range(cols):
            quadrant_names.append(chr(65 + r) + str(c + 1))
    return quadrant_names

def get_quadrants_and_sizes(x1, y1, x2, y2, cx1, cy1, cx2, cy2, rows, cols):
    quadrant_width = (cx2 - cx1) // cols
    quadrant_height = (cy2 - cy1) // rows
    quadrants = {}
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            col = (x - cx1) // quadrant_width
            row = (y - cy1) // quadrant_height
            col = min(col, cols - 1)
            row = min(row, rows - 1)
            quadrant = chr(65 + row) + str(col + 1)
            if quadrant not in quadrants:
                quadrants[quadrant] = 0
            quadrants[quadrant] += 1
    total_size = (x2 - x1 + 1) * (y2 - y1 + 1)
    for quadrant in quadrants:
        quadrants[quadrant] /= total_size
    return quadrants

def calculate_score(defects, rows, cols):
    score = 100
    center_quadrant = chr(65 + rows//2) + str(cols//2 + 1)
    quadrant_scores = {name: 0 for name in generate_quadrant_names(rows, cols)}
    processed_defects = set()
    for defect in defects:
        defect_id = (defect['class'], defect['coordinates']['x1'], defect['coordinates']['y1'], defect['coordinates']['x2'], defect['coordinates']['y2'])
        if defect_id not in processed_defects:
            penalty = penalties[defect['class']]
            for quadrant, size in defect['quadrants'].items():
                quadrant_penalty = penalty * size
                if quadrant == center_quadrant:
                    quadrant_penalty *= 2
                defect['penalty'] = quadrant_penalty
                score -= quadrant_penalty
                quadrant_scores[quadrant] += quadrant_penalty
            processed_defects.add(defect_id)
    for quadrant in quadrant_scores:
        quadrant_scores[quadrant] = max(100 - quadrant_scores[quadrant], 0)
    return max(score, 0), quadrant_scores

def put_text_with_background(image, text, position, font, font_scale, font_color, thickness, bg_color):
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = position
    cv2.rectangle(image, (x, y - text_height - 5), (x + text_width, y + 5), bg_color, -1)
    cv2.putText(image, text, (x, y), font, font_scale, font_color, thickness)

yaml_path = "configs/yolov8_config.yaml"
model_path = "path/to/yolov8_model.pt"
image_path = "path/to/your/image.jpg"
output_image_file = "results/output_image.jpg"
cropped_image_file = "results/cropped_image.jpg"
json_filename = "results/detection_results.json"

device = "cuda"  # Use "cpu" or "mps" for Apple Silicon

class_names = load_class_names(yaml_path)

model = YOLO(model_path)
model.to(device)

results = model(image_path)

detections = results[0].boxes

image = cv2.imread(image_path)
height, width, _ = image.shape

chapa_box = None
for box in detections:
    if int(box.cls[0]) == 0:
        chapa_box = box
        break

if chapa_box:
    cx1, cy1, cx2, cy2 = map(int, chapa_box.xyxy[0])
    margin_w = int((cx2 - cx1) * 0.01)
    margin_h = int((cy2 - cy1) * 0.01)
    cx1 = max(cx1 - margin_w, 0)
    cy1 = max(cy1 - margin_h, 0)
    cx2 = min(cx2 + margin_w, width)
    cy2 = min(cy2 + margin_h, height)
    crop_margin_w = int((cx2 - cx1) * 0.01)
    crop_margin_h = int((cy2 - cy1) * 0.01)
    crop_cx1 = max(cx1 - crop_margin_w, 0)
    crop_cy1 = max(cy1 - crop_margin_h, 0)
    crop_cx2 = min(cx2 + crop_margin_w, width)
    crop_cy2 = min(cy2 + crop_margin_h, height)
    
    results_summary = []
    errors_summary = []
    disregarded_summary = []

    for box in detections:
        cls = int(box.cls[0].item())
        class_name = class_names[cls]
        conf = float(box.conf[0].item())
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        if cls != 0:
            if conf >= 0.25:
                if cx1 <= x1 <= cx2 and cy1 <= y1 <= cy2 and cx1 <= x2 <= cx2 and cy1 <= y2 <= cy2:
                    quadrants_and_sizes = get_quadrants_and_sizes(x1, y1, x2, y2, cx1, cy1, cx2, cy2, 3, 3)
                    for quadrant, size in quadrants_and_sizes.items():
                        results_summary.append({
                            "quadrant": quadrant,
                            "class": class_name,
                            "confidence": conf,
                            "penalty": size * penalties[class_name],
                            "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                            "quadrants": quadrants_and_sizes
                        })
                else:
                    errors_summary.append({
                        "class": class_name,
                        "confidence": conf,
                        "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    })
            else:
                disregarded_summary.append({
                    "class": class_name,
                    "confidence": conf,
                    "coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                })

    score, quadrant_scores = calculate_score(results_summary, 3, 3)

    results_summary.sort(key=lambda x: x["quadrant"])

    print("\nParâmetros Utilizados:")
    parametros_utilizados = [
        ["Imagem", "datasets/granito-3nbcw/train/images/"],
        ["Nome do Arquivo", "2_1_032809_071_BRANCO-DALLAS_jpg.rf.2b910d1df77862191a4cedfd443a784e.jpg"],
        ["Linhas", 3],
        ["Colunas", 3],
        ["Margem", "1.0%"],
        ["Confiança mínima", 0.25],
        ["Margem de corte", "1.0%"]
    ]
    print(tabulate(parametros_utilizados, headers=["Parâmetro", "Valor"], tablefmt="grid"))

    print(f"Pontuação da chapa: {score:.2f}%\n")

    print("Resultados da detecção:")
    if results_summary:
        resultados_deteccao = [
            ["Quadrante", "Classe", "Confiança", "Penalidade", "Coordenadas"]
        ]
        for result in results_summary:
            penalty = result['penalty'] if 'penalty' in result else 0
            resultados_deteccao.append([
                result['quadrant'], result['class'], f"{result['confidence']:.2f}", f"{penalty:.2f}",
                f"({result['coordinates']['x1']}, {result['coordinates']['y1']}), ({result['coordinates']['x2']}, {result['coordinates']['y2']})"
            ])
        print(tabulate(resultados_deteccao, headers="firstrow", tablefmt="grid"))
    else:
        print("Nenhum defeito foi detectado na chapa.\n")

    if errors_summary:
        print("\nErros fora da chapa:")
        erros_fora_chapa = [
            ["Classe", "Confiança", "Coordenadas"]
        ]
        for error in errors_summary:
            erros_fora_chapa.append([
                error['class'], f"{error['confidence']:.2f}",
                f"({error['coordinates']['x1']}, {error['coordinates']['y1']}), ({error['coordinates']['x2']}, {error['coordinates']['y2']})"
            ])
        print(tabulate(erros_fora_chapa, headers="firstrow", tablefmt="grid"))
    else:
        print("Nenhum erro fora da chapa foi detectado.\n")

    if disregarded_summary:
        print("\nDefeitos desconsiderados por confiança mínima:")
        defeitos_desconsiderados = [
            ["Classe", "Confiança", "Coordenadas"]
        ]
        for disregarded in disregarded_summary:
            defeitos_desconsiderados.append([
                disregarded['class'], f"{disregarded['confidence']:.2f}",
                f"({disregarded['coordinates']['x1']}, {disregarded['coordinates']['y1']}), ({disregarded['coordinates']['x2']}, {disregarded['coordinates']['y2']})"
            ])
        print(tabulate(defeitos_desconsiderados, headers="firstrow", tablefmt="grid"))
    else:
        print("Nenhum defeito foi desconsiderado por confiança mínima.\n")

    output_data = {
        "2_1_032809_071_BRANCO-DALLAS_jpg.rf.2b910d1df77862191a4cedfd443a784e.jpg": {
            "results": results_summary,
            "score": f"{score:.2f}%",
            "output_image": output_image_file,
            "cropped_image": cropped_image_file
        }
    }

    if os.path.exists(json_filename):
        with open(json_filename, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data["2_1_032809_071_BRANCO-DALLAS_jpg.rf.2b910d1df77862191a4cedfd443a784e.jpg"] = output_data["2_1_032809_071_BRANCO-DALLAS_jpg.rf.2b910d1df77862191a4cedfd443a784e.jpg"]
        output_data = existing_data

    with open(json_filename, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Resultados salvos em '{json_filename}'\n")

    result_image = results[0].plot()

    cv2.imwrite(output_image_file, result_image)
    print(f"Imagem salva em: {output_image_file}\n")

    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    cropped_image = image[crop_cy1:crop_cy2, crop_cx1:crop_cx2]

    quadrant_width = (cx2 - cx1) // 3
    quadrant_height = (cy2 - cy1) // 3
    for i in range(1, 3):
        cv2.line(cropped_image, (cx1 + i * quadrant_width - crop_cx1, cy1 - crop_cy1), (cx1 + i * quadrant_width - crop_cx1, cy2 - crop_cy1), (0, 255, 0), 2)
    for i in range(1, 3):
        cv2.line(cropped_image, (cx1 - crop_cx1, cy1 + i * quadrant_height - crop_cy1), (cx2 - crop_cx1, cy1 + i * quadrant_height - crop_cy1), (0, 255, 0), 2)

    quadrant_names = generate_quadrant_names(3, 3)
    for i, name in enumerate(quadrant_names):
        qx = cx1 + (i % 3) * quadrant_width + quadrant_width // 2 - crop_cx1
        qy = cy1 + (i // 3) * quadrant_height + quadrant_height // 2 - crop_cy1
        put_text_with_background(cropped_image, name, (qx, qy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, (0, 0, 0))
        put_text_with_background(cropped_image, f"{quadrant_scores[name]:.2f}%", (qx, qy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, (0, 0, 0))

    for result in results_summary:
        x1 = result['coordinates']['x1'] - crop_cx1
        y1 = result['coordinates']['y1'] - crop_cy1
        x2 = result['coordinates']['x2'] - crop_cx1
        y2 = result['coordinates']['y2'] - crop_cy1
        cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(cropped_image, result['class'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(cropped_image_file, cropped_image)
    print(f"Imagem salva em: {cropped_image_file}\n")

    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
else:
    print("Nenhuma chapa (classe 0) foi detectada na imagem.")
```
## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para mais detalhes.

