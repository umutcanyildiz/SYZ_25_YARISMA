import json
import os
from typing import List, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from PIL import Image
import glob

class CompetitionJSONGenerator:
    """
    Yarışma için model tahminlerini JSON formatında hazırlayan sınıf
    """
    
    def __init__(self, takim_adi: str, takim_id: str, aciklama: str = "", versiyon: str = "v1.0"):
        """
        JSON generator'ı başlat
        
        Args:
            takim_adi: Takım ismi
            takim_id: Sistem tarafından verilen takım numarası
            aciklama: Modelin amacı veya kısa açıklama (opsiyonel)
            versiyon: Tahmin dosyasının versiyonu (opsiyonel)
        """
        self.kunye = {
            "takim_adi": takim_adi,
            "takim_id": takim_id,
            "aciklama": aciklama,
            "versiyon": versiyon
        }
        self.tahminler = []
    
    def add_prediction(self, filename: str, stroke: int, stroke_type: int = 3):
        """
        Tek bir tahmin ekle
        
        Args:
            filename: Dosya adı (uzantısı ile birlikte, klasör yolu olmadan)
            stroke: İnme durumu (0: yok, 1: var) - int formatında
            stroke_type: İnme tipi (0: iskemik, 1: kanamalı, 3: belirsiz/bilinmiyor)
        """
        # Dosya adının sadece isim+uzantı olduğundan emin ol
        filename = os.path.basename(filename)
        
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
        """
        JSON formatında sonuç üret
        
        Returns:
            Dict: Yarışma formatında JSON dictionary
        """
        return {
            "kunye": self.kunye,
            "tahminler": self.tahminler
        }
    
    def save_json(self, output_path: str):
        """
        JSON'u dosyaya kaydet
        
        Args:
            output_path: Çıktı dosyası yolu
        """
        result = self.generate_json()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"JSON dosyası '{output_path}' olarak kaydedildi.")
        print(f"Toplam {len(self.tahminler)} tahmin kaydedildi.")
    
    def validate_predictions(self):
        """
        Tahminleri doğrula
        """
        errors = []
        
        for i, tahmin in enumerate(self.tahminler):
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
        
        if errors:
            print("UYARILAR:")
            for error in errors:
                print(f"- {error}")
        else:
            print("Tüm tahminler geçerli!")
        
        return len(errors) == 0


# -----------------------------
# StrokeViT Model
# -----------------------------
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
    """
    StrokeViT modeli ile yarışma için tahmin yap ve JSON olarak kaydet
    """
    print(f"🚀 StrokeViT ile yarışma tahminleri başlatılıyor...")
    print(f"Model: {model_path}")
    print(f"Test Data: {test_data_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Transform
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
    print("✅ Model yüklendi")
    
    # JSON generator'ı başlat
    generator = CompetitionJSONGenerator(
        takim_adi=takim_adi,
        takim_id=takim_id,
        aciklama="StrokeViT Hybrid Model (ResNet50 + ViT)",
        versiyon="v1.0"
    )
    
    # Test dosyalarını bul
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
    
    with torch.no_grad():
        for i in range(0, len(all_files), batch_size):
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
                    print(f"⚠️  Dosya okuma hatası {file_path}: {e}")
                    continue
            
            if len(batch_images) == 0:
                continue
                
            # Batch'i hazırla ve tahmin yap
            batch_tensor = torch.stack(batch_images).to(device)
            outputs = model(batch_tensor)
            raw_probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            # MODEL TERSİNE EĞİTİLDİĞİ İÇİN PROBABİLİTY'Yİ TERSİNE ÇEVİR
            probs = 1 - raw_probs
            
            # DEBUG: İlk batch'te detay göster
            if i == 0:
                print(f"\n🔍 İlk {len(probs)} tahmin (model tersine eğitilmiş):")
                for j, (name, raw_p, final_p) in enumerate(zip(batch_names, raw_probs, probs)):
                    pred = 1 if final_p >= threshold else 0
                    print(f"  {name}: raw={raw_p:.4f} -> final={final_p:.4f} -> {'STROKE' if pred==1 else 'NO_STROKE'}")
            
            all_predictions.extend(probs)
            all_filenames.extend(batch_names)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"📊 İşlenen: {min(i + batch_size, len(all_files))}/{len(all_files)}")
    
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
    for filename, binary_pred in zip(all_filenames, binary_predictions):
        generator.add_prediction(filename, int(binary_pred), 3)
    
    # Doğrula ve kaydet
    if generator.validate_predictions():
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
    """
    Ana fonksiyon
    """
    print("=" * 60)
    print("STROKEViT YARIŞMA JSON HAZIRLAYICI - SON VERSİYON")
    print("=" * 60)
    
    print("KULLANIM:")
    predict_for_competition(
        model_path="/home/comp5/ARTEK/SYZ_25_YARISMA/BIRINCI_GOREV/Morpheus/resnet50_strokevit_small/strokevit_best_model_small_resnet.pth",
        test_data_path="/home/comp5/ARTEK/SYZ_25_YARISMA/deneme_png",
        takim_adi="Morpheus",
        takim_id="568784", 
        output_json_path="final_competition_predictions.json",
        threshold=0.3,
        batch_size=8
    )


if __name__ == "__main__":
    main()