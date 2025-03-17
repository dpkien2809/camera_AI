import faiss
import threading
import time
from datetime import datetime
import timm
import numpy as np
from PIL import Image
import paho.mqtt.client as mqtt
import json
import cv2
import base64

import torch
#import torch.nn.functional as F
from torchvision import transforms
from collections import deque
from LATransformer.model import LATransformerTest

# MQTT Configuration (No changes here)
BROKER = "103.110.87.199"
PORT = 1883
TOPIC = "/topic/detected/V21441M384"
USERNAME = "humbleHanet"
PASSWORD = "humbleHanet"

stored_embeddings = {}
similarity_threshold = 5250

# Load ViT
vit_base = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=751)
vit_base= vit_base

# Create La-Transformer
model = LATransformerTest(vit_base, lmbd=8)

# Load LA-Transformer
name = "la_with_lmbd_8"
save_path = "./net_best.pth"
model.load_state_dict(torch.load(save_path), strict=False)
model.eval()

# Store processed images
processed_images = []
processed_images = deque(maxlen=1000)

# Setup Faiss for storing instances
dimension = 10752  
index = faiss.IndexFlatIP(dimension)
index_with_ids = faiss.IndexIDMap(index)

# Preprocesing image
transform_query_list = [
    transforms.Resize((224,224), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
transform_gallery_list = [
    transforms.Resize(size=(224,224),interpolation=3), #Image.BICUBIC
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]
data_transforms = {
'query': transforms.Compose( transform_query_list ),
'gallery': transforms.Compose(transform_gallery_list),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust the face embedding extraction function
def get_img_embedding(img):
    """
    Extract the feature/embedding from the image using LA-Transformer.
    """
    img_tensor = data_transforms['query'](Image.fromarray(img)).unsqueeze(0).to(device)  
    with torch.no_grad():
        embedding = model(img_tensor)  
    
    return embedding.cpu().numpy().flatten()  

# Add instance to Faiss list if not detected
def add_embedding_to_faiss(embedding, timestamp):
    embedding_np = np.array(embedding).astype(np.float32)
    id_np = np.array([timestamp], dtype=np.int64)
    index_with_ids.add_with_ids(embedding_np.reshape(1, -1), id_np)  
    stored_embeddings[timestamp] = embedding

# Remove duplicate
def is_duplicate(image, timestamp):
    embedding = get_img_embedding(image)
    
    add_embedding_to_faiss(embedding, timestamp)
    
    embedding_np = np.array(embedding).astype(np.float32)
    k = 1  
    distances, indices = index_with_ids.search(embedding_np.reshape(1, -1), k) 

    if distances[0][0] > similarity_threshold: 
        print(f"[INFO] Duplicate detected with similarity: {distances[0][0]:.2f}")
        return True
    return False

# Read image
def process_image(image_b64, timestamp):
    try:
        header, data = image_b64.split(',', 1)
        image_data = base64.b64decode(data)
        np_array = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            raise ValueError("Decoded image is empty or invalid!")
        
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if not is_duplicate(image_rgb, timestamp):
            print(f"[INFO] New image received at {timestamp}, processing...")
            processed_images.append((timestamp, img))
        else:
            print(f"[INFO] Duplicate image skipped at {timestamp}")
    except Exception as e:
        print(f"[ERROR] Failed to process image: {e}")

# MQTT callback function
def on_message(client, userdata, message):
    payload = json.loads(message.payload.decode("utf-8"))
    timestamp = payload.get("date_time", str(datetime.now()))
    image_b64 = payload.get("image")
    
    if image_b64:
        print(f"[DEBUG] Base64 Image Length: {len(image_b64)}")
        process_image(image_b64, timestamp)
    else:
        print("[ERROR] No image data found in the message!")

# Initialize MQTT client
def mqtt_listener():
    client = mqtt.Client()
    client.username_pw_set(USERNAME, PASSWORD)
    client.on_message = on_message

    client.connect(BROKER, PORT, 60)
    client.subscribe(TOPIC)
    
    print("[INFO] Listening for MQTT messages...")
    client.loop_forever()

def move_and_cleanup_images():
    try:
        while True:
            current_time = datetime.now().timestamp()
              
            # Delete instance if it in list for more than 50 min (3000s)
            while processed_images and (
                current_time - 
                (processed_images[0][0] if isinstance(processed_images[0][0], (int, float)) 
                 else datetime.strptime(processed_images[0][0], "%Y-%m-%d %H:%M:%S").timestamp()) >= 3000):
                removed_image = processed_images.popleft()
  
                print(f"[INFO] Removed old image with timestamp: {removed_image[0]}")
                print(f"[INFO] Processed images count: {len(processed_images)}")
  
            time.sleep(10)  
    except Exception as e:
        print(f"[ERROR] Failed to clean up processed images: {e}")  



# Create thread
mqtt_thread = threading.Thread(target=mqtt_listener, daemon=True)  # For get image
cleanup_thread = threading.Thread(target=move_and_cleanup_images, daemon=True)  # For clean up image list

mqtt_thread.start()
cleanup_thread.start()

# Keep file running
while True:
    time.sleep(1)

# save image to hard disk
# timestamp, img = processed_images[0]
# print(f"Timestamp: {timestamp}")
# cv2.imwrite(f"./{timestamp}.jpg", img)


