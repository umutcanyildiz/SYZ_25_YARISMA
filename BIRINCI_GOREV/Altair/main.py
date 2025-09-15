import json
import os
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models # 'models' import edildi
from PIL import Image
import glob
from tqdm import tqdm
import time


class CompetitionJSONGenerator:
    def __init__(self, takim_adi: str, takim_id: str, aciklama: str = "", versiyon: str = "v1.0"):
        self.kunye = {
            "takim_adi": takim_adi,
            "takim_id": takim_id,
            "aciklama": aciklama,
            "versiyon": versiyon
        }
        self.tahminler = []

    def add_prediction(self, filename: str, stroke: int, stroke_type: int = 3):
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0] + ".dcm"
        if isinstance(stroke, (float, np.floating)):
            stroke = int(stroke)
        elif isinstance(stroke, str):
            stroke = int(float(stroke))
        if isinstance(stroke_type, (float, np.floating)):
            stroke_type = int(stroke_type)
        elif isinstance(stroke_type, str):
            stroke_type = int(float(stroke_type))
        prediction = {
            "filename": filename,
            "stroke": stroke,
            "stroke_type": stroke_type
        }
        self.tahminler.append(prediction)

    def generate_json(self) -> Dict[str, Any]:
        return {
            "kunye": self.kunye,
            "tahminler": self.tahminler
        }

    def save_json(self, output_path: str):
        result = self.generate_json()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"JSON dosyası '{output_path}' olarak kaydedildi.")
        print(f"Toplam {len(self.tahminler)} tahmin kaydedildi.")

    def validate_predictions(self):
        errors = []
        print("🔍 Tahminler doğrulanıyor...")
        for i in tqdm(range(len(self.tahminler)), desc="🔍 Doğrulama", unit="tahmin"):
            tahmin = self.tahminler[i]
            if not tahmin['filename'] or not isinstance(tahmin['filename'], str):
                errors.append(f"Tahmin {i}: Geçersiz dosya adı")
            if tahmin['stroke'] not in [0, 1]:
                errors.append(f"Tahmin {i}: stroke değeri 0 veya 1 olmalı, mevcut: {tahmin['stroke']}")
            if not isinstance(tahmin['stroke'], int):
                errors.append(f"Tahmin {i}: stroke değeri int formatında olmalı, mevcut tip: {type(tahmin['stroke'])}")
            if tahmin['stroke_type'] not in [0, 1, 3]:
                errors.append(f"Tahmin {i}: stroke_type değeri 0, 1 veya 3 olmalı, mevcut: {tahmin['stroke_type']}")
            if not isinstance(tahmin['stroke_type'], int):
                errors.append(f"Tahmin {i}: stroke_type değeri int formatında olmalı, mevcut tip: {type(tahmin['stroke_type'])}")
            time.sleep(0.001)  # Doğrulama çubuğu da görünsün
                
        if errors:
            print("UYARILAR:")
            for error in errors:
                print(f"- {error}")
        else:
            print("Tüm tahminler geçerli!")
        return len(errors) == 0


def predict_for_competition(model_path: str, test_data_path: str, takim_adi: str, takim_id: str,
                            output_json_path: str, threshold: float = 0.5):
    
    print(f"🚀 Tahminler başlatılıyor...")
    print(f"Model Mimarisi: ResNet-34")
    print(f"Model Dosyası: {model_path}")
    print(f"Test Veri Yolu: {test_data_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")

    IMG_SIZE = 224
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]
    
    print("📦 Transform ve model hazırlanıyor...")
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), 
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    model = models.resnet34(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 1) 
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model yüklendi ve değerlendirme moduna alındı.")

    # JSON Generator'ı başlat (Bu sınıfın kodun başka bir yerinde tanımlı olduğunu varsayıyoruz)
    generator = CompetitionJSONGenerator(
        takim_adi=takim_adi,
        takim_id=takim_id,
        aciklama="1.Gorev - Morpheus Takımı Tahminleri (ResNet-34 Mimarisi - Ters Olasılık Düzeltmeli)",
        versiyon="v1.3"
    )

    print("🔍 PNG dosyaları aranıyor...")
    all_files = sorted(glob.glob(os.path.join(test_data_path, '**', '*.png'), recursive=True))
    print(f"📁 {len(all_files)} adet PNG dosyası bulundu.")
    
    if not all_files:
        print("❌ Hiç PNG dosyası bulunamadı! Lütfen test verisi yolunu kontrol edin.")
        return

    all_predictions = []
    all_filenames = []
    
    print("-" * 70)
    print(">>> BİREYSEL RESİM TAHMİN SÜRECİ BAŞLATILIYOR <<<")
    print("(Not: Modelin 'tersine' öğrenmiş olma ihtimaline karşı olasılıklar düzeltilmektedir.)")
    print("-" * 70)

    with torch.no_grad():
        for file_path in tqdm(all_files, desc="Resimler İşleniyor", unit="resim"):
            filename = os.path.basename(file_path)
            
            try:
                img = Image.open(file_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(device)
            except Exception as e:
                tqdm.write(f"⚠️  {filename}: Dosya okunamadı veya dönüştürülemedi! Hata: {e}")
                continue

            outputs = model(input_tensor)
            
            prob_no_stroke = torch.sigmoid(outputs).item() 
            prob = 1.0 - prob_no_stroke
            
            prediction = 1 if prob >= threshold else 0
            result_text = "STROKE" if prediction == 1 else "NO_STROKE"

            # tqdm.write kullanarak çubuk bozulmasın
            tqdm.write(f"-> {filename:<30} | Olasılık (Stroke): {prob:.4f} | Sonuç: {result_text}")
            
            # Yavaşlatma - çubuğun görünmesi için
            time.sleep(0.05)

            all_predictions.append(prob) # Düzeltilmiş olasılığı listeye ekle
            all_filenames.append(filename)

    print("-" * 70)
    print(f"✅ Bireysel tahmin süreci tamamlandı.")
    print("-" * 70)

    probs_array = np.array(all_predictions)
    print("\n🔍 Genel İstatistikler:")
    print(f"  Threshold ({threshold}) değeri ve üstü tahmin sayısı: {(probs_array >= threshold).sum()}/{len(probs_array)}")

    binary_predictions = (probs_array >= threshold).astype(int)
    
    print("📝 JSON için tahminler ekleniyor...")
    for filename, binary_pred in tqdm(zip(all_filenames, binary_predictions), desc="📝 JSON Hazırlanıyor", total=len(all_filenames), unit="kayıt"):
        generator.add_prediction(filename, int(binary_pred), 3)
        time.sleep(0.01)  # JSON hazırlama da görünsün

    if generator.validate_predictions():
        print("💾 JSON dosyası kaydediliyor...")
        generator.save_json(output_json_path)
        
        stroke_count = sum(1 for p in generator.tahminler if p['stroke'] == 1)
        no_stroke_count = len(generator.tahminler) - stroke_count
        print(f"📈 FİNAL SONUÇ: {stroke_count} STROKE (+), {no_stroke_count} NO STROKE (-)")
        print(f"🎯 Kullanılan threshold: {threshold}")
            
    else:
        print("❌ JSON doğrulama başarısız oldu! Çıktı dosyası oluşturulmadı.")


def main():
    print("=" * 60)
    print("ResNet-34 Mimarisi ile Yarışma Tahmin Betiği")
    print("=" * 60)

    predict_for_competition(
        model_path="/home/comp5/ARTEK/SYZ_25_YARISMA/BIRINCI_GOREV/Altair/agirlik_model/best_resnet34_model.pth", 
        test_data_path="/home/comp5/ARTEK/SYZ_25_YARISMA/BIRINCI_GOREV/ornek_veriler/deneme_png",
        takim_adi="Morpheus",
        takim_id="657266",
        output_json_path="yarisma_ciktisi.json",
        threshold=0.3, 
    )


if __name__ == "__main__":
    main()