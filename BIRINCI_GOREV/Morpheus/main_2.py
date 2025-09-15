import json
import os
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm #type: ignore
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

        # Dosya adının sadece isim+uzantı olduğundan emin ol
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0] + ".dcm"
        
        # stroke değerinin int olduğundan emin ol
        if isinstance(stroke, (float, np.floating)):
            stroke = int(stroke)
        elif isinstance(stroke, str):
            stroke = int(float(stroke))
        
        # stroke_type değerinin int olduğundan emin ol
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
            # Dosya adı kontrolü
            if not tahmin['filename'] or not isinstance(tahmin['filename'], str):
                errors.append(f"Tahmin {i}: Geçersiz dosya adı")
            
            # Stroke değeri kontrolü
            if tahmin['stroke'] not in [0, 1]:
                errors.append(f"Tahmin {i}: stroke değeri 0 veya 1 olmalı, mevcut: {tahmin['stroke']}")
            
            # Stroke değerinin int olup olmadığını kontrol et
            if not isinstance(tahmin['stroke'], int):
                errors.append(f"Tahmin {i}: stroke değeri int formatında olmalı, mevcut tip: {type(tahmin['stroke'])}")
            
            # Stroke_type değeri kontrolü
            if tahmin['stroke_type'] not in [0, 1, 3]:
                errors.append(f"Tahmin {i}: stroke_type değeri 0, 1 veya 3 olmalı, mevcut: {tahmin['stroke_type']}")
            
            # Stroke_type değerinin int olup olmadığını kontrol et
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


class StrokeViT(nn.Module):
    def __init__(self, num_classes=1):
        super(StrokeViT, self).__init__()
        self.resnet = timm.create_model("resnet50", pretrained=True, num_classes=0, global_pool="avg")
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0, global_pool="")
        self.fc = nn.Linear(2048 + self.vit.embed_dim, num_classes)

    def forward(self, x):
        res_feat = self.resnet(x)
        vit_feat = self.vit.forward_features(x)

        if vit_feat.dim() == 4:       # [B, C, H, W]
            vit_feat = torch.flatten(vit_feat, 1)
        elif vit_feat.dim() == 3:     # [B, N, C]
            vit_feat = vit_feat.mean(dim=1)

        combined = torch.cat((res_feat, vit_feat), dim=1)
        out = self.fc(combined)
        return out


def predict_for_competition(model_path: str, test_data_path: str, takim_adi: str, takim_id: str, 
                          output_json_path: str, threshold: float = 0.3, batch_size: int = 8):

    print(f"🚀 Tahminler başlatılıyor...")
    print(f"Model Mimarisi: StrokeViT (ResNet50 + ViT)")
    print(f"Model Dosyası: {model_path}")
    print(f"Test Veri Yolu: {test_data_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan Cihaz: {device}")
    
    # Transform
    print("📦 Transform ve model hazırlanıyor...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Modeli yükle
    model = StrokeViT(num_classes=1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("✅ Model yüklendi ve değerlendirme moduna alındı.")
    
    # JSON generator'ı başlat
    generator = CompetitionJSONGenerator(
        takim_adi=takim_adi,
        takim_id=takim_id,
        aciklama="1.Gorev - Morpheus Takımı Tahminleri",
        versiyon="v1.0"
    )
    
    # Test dosyalarını bul
    print("🔍 PNG dosyaları aranıyor...")
    all_files = glob.glob(os.path.join(test_data_path, '*.png'))
    if not all_files:
        all_files = glob.glob(os.path.join(test_data_path, '**', '*.png'), recursive=True)
    
    all_files = sorted(all_files)
    print(f"📁 {len(all_files)} PNG dosyası bulundu")
    
    if len(all_files) == 0:
        print("❌ Hiç PNG dosyası bulunamadı!")
        return
    
    # İlk birkaç dosya adını göster
    print("📄 Bulunan dosyalardan örnekler:")
    for i, f in enumerate(all_files[:5]):
        print(f"  {i+1}. {os.path.basename(f)}")
    if len(all_files) > 5:
        print(f"  ... ve {len(all_files)-5} dosya daha")
    
    # Tahminleri topla
    all_predictions = []
    all_filenames = []
    
    print("-" * 70)
    print(">>> BATCH TAHMİN SÜRECİ BAŞLATILIYOR <<<")
    print(f"Batch Size: {batch_size}")
    print("-" * 70)
    
    num_batches = (len(all_files) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, len(all_files), batch_size), desc="Batch İşleniyor", unit="batch", total=num_batches):
            batch_files = all_files[i:i+batch_size]
            batch_images = []
            batch_names = []
            
            for file_path in batch_files:
                try:
                    img = Image.open(file_path).convert('RGB')
                    img_tensor = transform(img)
                    batch_images.append(img_tensor)
                    batch_names.append(os.path.basename(file_path))
                except Exception as e:
                    tqdm.write(f"⚠️  Dosya okuma hatası {file_path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
                
            batch_tensor = torch.stack(batch_images).to(device)
            outputs = model(batch_tensor)
            raw_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            probs = 1 - raw_probs
            
            if i == 0:
                tqdm.write(f"\n🔍 İlk {len(probs)} tahmin (model tersine eğitilmiş):")
                for j, (name, raw_p, final_p) in enumerate(zip(batch_names, raw_probs, probs)):
                    pred = 1 if final_p >= threshold else 0
                    tqdm.write(f"  {name}: raw={raw_p:.4f} -> final={final_p:.4f} -> {'STROKE' if pred==1 else 'NO_STROKE'}")
            
            all_predictions.extend(probs)
            all_filenames.extend(batch_names)
            
            # Yavaşlatma - batch çubuğunu görmek için
            time.sleep(0.02)
    
    print("-" * 70)
    print(f"✅ Batch tahmin süreci tamamlandı.")
    print("-" * 70)
    
    print(f"✅ Toplam {len(all_predictions)} tahmin tamamlandı")
    
    # Probability istatistikleri
    probs_array = np.array(all_predictions)
    print(f"\n🔍 Final Probability İstatistikleri (tersine çevrilmiş):")
    print(f"  Min: {probs_array.min():.4f}")
    print(f"  Max: {probs_array.max():.4f}")
    print(f"  Mean: {probs_array.mean():.4f}")
    print(f"  Threshold ({threshold}) üstü: {(probs_array >= threshold).sum()}/{len(probs_array)}")
    
    # JSON'a ekle - DİREKT BINARY DÖNÜŞÜM YAP
    binary_predictions = (probs_array >= threshold).astype(int)
    
    print(f"\n🔍 Binary dönüşüm kontrolü:")
    print(f"  Threshold: {threshold}")
    print(f"  Binary 1 sayısı: {binary_predictions.sum()}/{len(binary_predictions)}")
    print(f"  İlk 10 binary sonuç: {binary_predictions[:10]}")
    
    # Manuel olarak ekle - load_from_model_output fonksiyonu sorunlu
    print("📝 JSON için tahminler ekleniyor...")
    for filename, binary_pred in tqdm(zip(all_filenames, binary_predictions), desc="📝 JSON Hazırlanıyor", total=len(all_filenames), unit="kayıt"):
        generator.add_prediction(filename, int(binary_pred), 3)
        time.sleep(0.01)  # JSON hazırlama da görünsün
    
    # Doğrula ve kaydet
    if generator.validate_predictions():
        print("💾 JSON dosyası kaydediliyor...")
        generator.save_json(output_json_path)
        print(f"🎉 Yarışma JSON dosyası başarıyla kaydedildi: {output_json_path}")
        
        # Özet bilgileri göster
        stroke_count = sum(1 for p in generator.tahminler if p['stroke'] == 1)
        no_stroke_count = len(generator.tahminler) - stroke_count
        print(f"📈 FINAL SONUÇ: {stroke_count} STROKE (+), {no_stroke_count} NO STROKE (-)")
        print(f"🎯 Kullanılan threshold: {threshold}")
        print(f"🔄 Model tersine eğitildiği için probability tersine çevrildi (1-prob)")
        
        # Farklı threshold'larla özet
        print(f"\n📊 Farklı threshold'larla sonuçlar:")
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            stroke_count_t = (probs_array >= t).sum()
            print(f"  Threshold {t:.1f}: {stroke_count_t} stroke pozitif")
        
        # JSON kontrolü - ilk birkaç tahmin
        print(f"\n📋 JSON'daki ilk 5 tahmin:")
        for i, tahmin in enumerate(generator.tahminler[:5]):
            print(f"  {tahmin['filename']}: stroke={tahmin['stroke']}")
            
    else:
        print("❌ Doğrulama başarısız!")


# Ana çalıştırma fonksiyonu
def main():
    print("=" * 60)
    print("StrokeViT Mimarisi ile Yarışma Tahmin Betiği")
    print("=" * 60)

    predict_for_competition(
        model_path="/home/comp5/ARTEK/SYZ_25_YARISMA/BIRINCI_GOREV/Morpheus/resnet50_strokevit_small/strokevit_best_model_small_resnet.pth",
        test_data_path="/home/comp5/ARTEK/SYZ_25_YARISMA/BIRINCI_GOREV/ornek_veriler/deneme_png",
        takim_adi="Morpheus",
        takim_id="657266", 
        output_json_path="yarisma_ciktisi.json",
        threshold=0.3,
        batch_size=8
    )


if __name__ == "__main__":
    main()