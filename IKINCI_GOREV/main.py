import os
import glob
import json
from PIL import Image
import torch
import timm
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from collections import Counter 

MODEL_ADI = 'tf_efficientnetv2_m.in21k'
MODEL_PATH = "efficientNetv2.pth"
TEST_DATA_DIR = "/home/comp5/ARTEK/SYZ_25_YARISMA/IKINCI_GOREV/MR_testset_PNG" 
OUTPUT_JSON_PATH = "yarisma_ciktisi.json" 
PREDICTION_THRESHOLD = 0.35 

CLASS_NAMES = ["HiperakutAkut", "NormalKronik", "Subakut"]
JSON_CLASS_MAP = {
    "HiperakutAkut": "hyperacute_acute",
    "NormalKronik": "normal_chronic",
    "Subakut": "subacute"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_model(model_name, num_classes, model_path):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def get_transforms(model):
    data_config = timm.data.resolve_data_config({}, model=model)
    img_size = data_config['input_size'][1]
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std'])
    ])
    return transform

def predict():
    print(f"🚀 Model yükleniyor: {MODEL_PATH}")
    model = create_model(MODEL_ADI, len(CLASS_NAMES), MODEL_PATH)
    transform = get_transforms(model)
    
    try:
        patient_folders = [d for d in os.listdir(TEST_DATA_DIR) if os.path.isdir(os.path.join(TEST_DATA_DIR, d))]
        if not patient_folders:
            print(f"❌ HATA: '{TEST_DATA_DIR}' klasöründe hasta klasörü (PID_...) bulunamadı.")
            return
    except FileNotFoundError:
        print(f"❌ HATA: Test verisi klasörü bulunamadı: '{TEST_DATA_DIR}'")
        return

    predictions_list = []

    for patient_id in tqdm(patient_folders, desc="🤖 Hastalar işleniyor"):
        patient_image_paths = glob.glob(os.path.join(TEST_DATA_DIR, patient_id, "*.png"))
        
        if not patient_image_paths:
            continue # Boş klasörleri atla

        all_predictions_for_patient = []

        for img_path in patient_image_paths:
            image = Image.open(img_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                output = model(image_tensor)
                pred_idx = torch.argmax(output, dim=1).item()
                predicted_class_name = CLASS_NAMES[pred_idx]
                all_predictions_for_patient.append(predicted_class_name)

        num_images = len(all_predictions_for_patient)
        prediction_counts = Counter(all_predictions_for_patient)
        
        patient_prediction = {"PatientID": patient_id}
        for class_name, json_key in JSON_CLASS_MAP.items():
            count = prediction_counts.get(class_name, 0)
            if (count / num_images) >= PREDICTION_THRESHOLD:
                patient_prediction[json_key] = 1
            else:
                patient_prediction[json_key] = 0
            
        predictions_list.append(patient_prediction)
        
    final_submission = {
        "kunye": {
            "takim_adi": "TUSEB_SYZ_MR",
            "takim_id": "0123456",
            "aciklama": "MR Tahmin Verileri v2.0",
            "versiyon": "v2.0"
        },
        "tahminler": predictions_list
    }

    with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_submission, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Tahminler tamamlandı ve '{OUTPUT_JSON_PATH}' dosyasına kaydedildi.")
    print(f"ℹ️ Not: Tahminler için {PREDICTION_THRESHOLD*100:.0f}% eşik değeri kullanıldı.")

if __name__ == "__main__":
    predict()