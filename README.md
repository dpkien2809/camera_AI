## MQTT-Based Image Processing with LA-Transformer
---
## Overview
This Python script performs image processing using the LA-Transformer model for detecting and handling duplicate images. The script listens for MQTT messages containing base64-encoded images, processes the images, and stores unique image embeddings using Faiss.

### Key Features:
1. **Image Processing:** Converts base64 images to RGB format.
2. **Duplicate Detection:** Uses embeddings to detect and skip duplicate images based on a similarity threshold.
3. **MQTT Integration:** Listens to incoming images over MQTT.
4. **Embeddings Storage:** Stores embeddings using Faiss for efficient nearest-neighbor search.
5. **Automatic Cleanup:** Removes old images from the processed queue after 50 minutes.

---

## Requirements
- Python 3.x
- Libraries: `torch`, `torchvision`, `numpy`, `faiss`, `Pillow`, `opencv-python`, `paho-mqtt`, `timm`

Install dependencies using:
```bash
pip install torch torchvision numpy faiss-cpu Pillow opencv-python paho-mqtt timm
```

---

## Code Explanation

### 1. **Model Setup**
- The script loads a pre-trained Vision Transformer (ViT) and integrates it into a LA-Transformer model.
- The LA-Transformer weights are loaded from a checkpoint file (`net_best.pth`).

### 2. **MQTT Configuration**
```python
BROKER = \"103.110.87.199\"
PORT = 1883
TOPIC = \"/topic/detected/V21441M384\"
USERNAME = \"humbleHanet\"
PASSWORD = \"humbleHanet\"
```
- MQTT Broker details are configured for message reception.

### 3. **Image Preprocessing**
- The script defines transformations for the input images:
  - Resize to 224x224.
  - Normalize using ImageNet statistics.

### 4. **Embedding Extraction**
```python
def get_img_embedding(img):
    img_tensor = data_transforms['query'](Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.cpu().numpy().flatten()
```
- Converts the image to a tensor and extracts embeddings using LA-Transformer.

### 5. **Duplicate Detection**
- Embeddings are stored in a Faiss index for nearest-neighbor search.
- Duplicate detection is based on a similarity threshold (5250).

### 6. **Image Processing Workflow**
#### MQTT Listener:
- Subscribes to the configured MQTT topic and processes incoming images.

#### Duplicate Check:
```python
if not is_duplicate(image_rgb, timestamp):
    print(f\"[INFO] New image received at {timestamp}, processing...\")
    processed_images.append((timestamp, img))
else:
    print(f\"[INFO] Duplicate image skipped at {timestamp}\")
```
- New images are added to the queue, while duplicates are skipped.

#### Cleanup Task:
- Removes images older than 50 minutes (3000 seconds).

---

## How to Run
1. Clone the repository and navigate to the project folder.
2. Ensure `net_best.pth` (model checkpoint) is available in the same directory.
3. Set up the environment using the following commands:
```bash
%cd /kaggle/working/
!git clone https://github.com/SiddhantKapil/LA-Transformer.git
%cd LA-Transformer
!conda install --quiet -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia 
!conda install --quiet -y timm faiss-gpu tqdm  
!pip install paho-MQTT
!pip install --upgrade Pillow
!pip install -q gdown
!gdown --folder https://drive.google.com/drive/folders/1sYwyS0DOIRp_yinZ0qMSmcpeWd6AmN9m?usp=drive_link -O /kaggle/working/LA-Transformer
```
4. Run the script:
```bash
python company_camera.py
```

---

## Threads
- **MQTT Listener:** Continuously listens for new images.
- **Cleanup Thread:** Periodically removes old images.

---

## Troubleshooting
- Ensure the MQTT broker is reachable and the topic is correctly configured.
- Verify that `net_best.pth` exists and matches the LA-Transformer architecture.
- Check Python dependencies using `pip list`.

---

## Example Output
```
[INFO] Listening for MQTT messages...
[INFO] New image received at 2024-12-14 12:00:00, processing...
[INFO] Duplicate image skipped at 2024-12-14 12:05:00
[INFO] Removed old image with timestamp: 2024-12-14 11:50:00
```
