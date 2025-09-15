import os
import pydicom
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
import warnings
import time

# --- ÖN İŞLEME FONKSİYONLARI ---
def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """CLAHE (Contrast Limited Adaptive Histogram Equalization) uygular."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def apply_gamma(img, gamma=1.2):
    """Gamma düzeltme uygular - güvenli versiyon."""
    try:
        # Float32'ye dönüştür
        img_float = img.astype('float32')
        
        # Boş veya bozuk görüntü kontrolü
        if img_float.size == 0:
            return img
        
        # Min-max normalize et (NaN ve inf kontrolü ile)
        min_val = np.nanmin(img_float)
        max_val = np.nanmax(img_float)
        
        # Eğer tüm piksel değerleri aynıysa, normalizasyon yapmadan döndür
        if max_val == min_val or np.isnan(min_val) or np.isnan(max_val) or np.isinf(min_val) or np.isinf(max_val):
            return img
        
        # Güvenli normalizasyon
        norm = (img_float - min_val) / (max_val - min_val)
        
        # NaN ve inf kontrolü
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 0-1 aralığında sınırla
        norm = np.clip(norm, 0.0, 1.0)
        
        # Gamma düzeltme (güvenli)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gamma_corrected = np.power(norm, gamma)
        
        # NaN kontrolü
        gamma_corrected = np.nan_to_num(gamma_corrected, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 0-255 aralığına dönüştür
        result = (gamma_corrected * 255.0).clip(0, 255).astype('uint8')
        
        return result
        
    except Exception:
        # Hata durumunda orijinal görüntüyü döndür
        return img

def auto_crop(img):
    """
    Görüntüdeki en büyük nesneyi (konturu) bularak etrafındaki boş siyah alanları kırpar.
    Geliştirilmiş ve güvenli versiyon.
    """
    try:
        # Boş görüntü kontrolü
        if img is None or img.size == 0:
            return img
        
        # Görüntünün tek kanallı (grayscale) olduğundan emin olalım
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # Minimum boyut kontrolü
        if gray.shape[0] < 50 or gray.shape[1] < 50:
            return img

        # Gürültüyü azaltmak için hafif bir bulanıklaştırma
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Eşikleme ile nesneyi arka plandan ayır
        # Eşik değeri (10), çok koyu piksellerin arka plan olarak kabul edilmesini sağlar
        _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)

        # Görüntüdeki tüm dış konturları bul
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Eğer hiç kontur bulunamazsa orijinal resmi geri döndür
        if not contours:
            return img

        # En büyük alana sahip konturu ana nesne olarak seç
        c = max(contours, key=cv2.contourArea)
        
        # Kontur alanı kontrolü
        area = cv2.contourArea(c)
        if area < 100:  # Çok küçük kontur
            return img

        # En büyük konturun sınırlayıcı kutusunu (bounding box) bul
        x, y, w, h = cv2.boundingRect(c)

        # Çok küçük nesne tespiti - eğer nesne çok küçükse kırpma yapma
        if w < 50 or h < 50:
            return img

        # Kenarlara çok yakın kırpmamak için küçük bir pay (margin) bırakalım
        margin_x = max(1, int(w * 0.04))  # Genişliğin %4'ü, minimum 1
        margin_y = max(1, int(h * 0.05))  # Yüksekliğin %5'i, minimum 1

        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(img.shape[1], x + w + margin_x)
        y_end = min(img.shape[0], y + h + margin_y)

        # Geçerli koordinat kontrolü
        if x_end <= x_start or y_end <= y_start:
            return img

        # Kırpılmış görüntüyü oluştur
        cropped = img[y_start:y_end, x_start:x_end]

        # Eğer kırpma sonucu çok küçük bir alan kalırsa veya boş ise orijinali kullan
        if cropped.size == 0 or cropped.shape[0] < 100 or cropped.shape[1] < 100:
            return img

        return cropped
        
    except Exception as e:
        # Hata durumunda orijinal görüntüyü döndür
        print(f"⚠️  Auto crop hatası: {e}, orijinal görüntü kullanılıyor")
        return img

def get_window_values(dicom_resim, pixel_array):
    """DICOM'dan window center ve width değerlerini alır. Eksikse fallback yapar."""
    try:
        wc = dicom_resim.WindowCenter
        ww = dicom_resim.WindowWidth
        if isinstance(wc, pydicom.multival.MultiValue):
            wc = float(wc[0])
        else:
            wc = float(wc)
        if isinstance(ww, pydicom.multival.MultiValue):
            ww = float(ww[0])
        else:
            ww = float(ww)
    except Exception:
        wc = float(np.mean(pixel_array))
        ww = float(np.max(pixel_array) - np.min(pixel_array))
    return wc, ww

def process_batch(batch_files, enable_clahe=True, enable_gamma=True, gamma_value=1.2, 
                 clahe_clip_limit=2.0, clahe_tile_size=(8, 8)):
    """
    Bir batch dosyayı işler ve ön işleme adımlarını uygular.
    """
    results = []
    for dicom_path, output_path in batch_files:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dicom_resim = pydicom.dcmread(dicom_path, force=True)

            if not hasattr(dicom_resim, "PixelData"):
                with open("failed_files.txt", "a") as f:
                    f.write(f"{dicom_path} -> PixelData yok\n")
                results.append(False)
                continue

            pixel_array = dicom_resim.pixel_array.astype(np.float32)
            slope = float(getattr(dicom_resim, "RescaleSlope", 1.0))
            intercept = float(getattr(dicom_resim, "RescaleIntercept", 0.0))
            hu_image = pixel_array * slope + intercept

            wc, ww = get_window_values(dicom_resim, hu_image)
            min_val = wc - ww / 2
            max_val = wc + ww / 2
            if max_val == min_val:
                max_val = min_val + 1

            img_windowed = np.clip(hu_image, min_val, max_val)

            img_norm = ((img_windowed - min_val) / (max_val - min_val)) * 255.0
            img_uint8 = img_norm.astype(np.uint8)

            # --- ÖN İŞLEME ADIMLARI ---
            processed_img = img_uint8.copy()
            
            # 1. CLAHE uygula (kontrast iyileştirme)
            if enable_clahe:
                processed_img = apply_clahe(processed_img, 
                                          clip_limit=clahe_clip_limit, 
                                          tile_grid_size=clahe_tile_size)
            
            # 2. Gamma düzeltme uygula
            if enable_gamma:
                processed_img = apply_gamma(processed_img, gamma=gamma_value)
            
            # 3. Otomatik kırpma uygula
            processed_img = auto_crop(processed_img)
            # --- ÖN İŞLEME BİTİŞ ---

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # İşlenmiş görüntüyü PNG olarak kaydet
            saved = cv2.imwrite(output_path, processed_img)
            if not saved:
                with open("failed_files.txt", "a") as f:
                    f.write(f"{dicom_path} -> PNG kaydedilemedi\n")
                results.append(False)
            else:
                results.append(True)

        except Exception as e:
            with open("failed_files.txt", "a") as f:
                f.write(f"{dicom_path} -> {str(e)}\n")
            results.append(False)

    return results

def convert_dicom_directory(input_directory, output_directory, max_workers=6, batch_size=256,
                          enable_clahe=True, enable_gamma=True, gamma_value=1.2,
                          clahe_clip_limit=2.0, clahe_tile_size=(8, 8)):
    """
    DICOM dosyalarını PNG'ye dönüştürür ve ön işleme uygular.
    
    Args:
        input_directory: DICOM dosyalarının bulunduğu klasör
        output_directory: PNG dosyalarının kaydedileceği klasör
        max_workers: Paralel işlem sayısı
        batch_size: Batch boyutu
        enable_clahe: CLAHE ön işlemesini etkinleştir
        enable_gamma: Gamma düzeltmeyi etkinleştir
        gamma_value: Gamma değeri (1.0 = değişiklik yok, >1.0 = daha parlak, <1.0 = daha koyu)
        clahe_clip_limit: CLAHE clip limit değeri
        clahe_tile_size: CLAHE tile boyutu (genişlik, yükseklik)
    """
    console = Console()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Başarısız dosyalar logunu temizle
    if os.path.exists("failed_files.txt"):
        os.remove("failed_files.txt")

    conversion_tasks = []
    total_files = 0

    console.print("[yellow]DICOM dosyaları taranıyor...[/yellow]")
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith('.dcm'):
                total_files += 1
                dicom_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, input_directory)
                output_subdir = os.path.join(output_directory, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, f"{os.path.splitext(file)[0]}.png")
                conversion_tasks.append((dicom_path, output_path))

    if total_files == 0:
        console.print("[red]Hiç DICOM dosyası bulunamadı![/red]")
        return

    batches = [conversion_tasks[i:i + batch_size] for i in range(0, len(conversion_tasks), batch_size)]
    
    start_time = time.time()

    # Ön işleme ayarlarını göster
    console.print("\n[bold blue]Ön İşleme Ayarları:[/bold blue]")
    console.print(f"CLAHE: {'✓' if enable_clahe else '✗'} (Clip Limit: {clahe_clip_limit}, Tile Size: {clahe_tile_size})")
    console.print(f"Gamma Düzeltme: {'✓' if enable_gamma else '✗'} (Gamma: {gamma_value})")
    console.print(f"Otomatik Kırpma: ✓")
    console.print("")

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Dönüştürülüyor ve ön işleme uygulanıyor...", total=total_files)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch, enable_clahe, enable_gamma, gamma_value,
                              clahe_clip_limit, clahe_tile_size): batch
                for batch in batches
            }

            for future in as_completed(future_to_batch):
                batch_results = future.result()
                completed += len(batch_results)
                progress.update(task, completed=completed)

    end_time = time.time()
    total_time = end_time - start_time
    average_time = total_time / total_files if total_files > 0 else 0
    
    console.print("\n[bold green]İşlem Tamamlandı![/bold green]")
    console.print("---" * 15)
    console.print(f"Toplam dönüştürülen resim sayısı: {total_files}")
    console.print(f"Toplam geçen süre: {total_time:.2f} saniye")
    console.print(f"Bir resmi ortalama dönüştürme süresi: {average_time:.4f} saniye")
    console.print(f"Paralel işlem sayısı (workers): {max_workers}")
    console.print("---" * 15)
    if os.path.exists("failed_files.txt"):
        console.print("[yellow]Başarısız dosyalar 'failed_files.txt' içine kaydedildi.[/yellow]")
    else:
        console.print("[green]Tüm dosyalar başarıyla işlendi![/green]")

if __name__ == '__main__':
    input_dir = "/home/comp5/ARTEK/SYZ_25_YARISMA/BIRINCI_GOREV/ornek_veriler/deneme_dcom"
    output_dir = "/home/comp5/ARTEK/SYZ_25_YARISMA/BIRINCI_GOREV/ornek_veriler/deneme_png"

    # Gelişmiş ön işleme ile dönüştürme
    convert_dicom_directory(
        input_dir, 
        output_dir, 
        max_workers=16, 
        batch_size=128,
        enable_clahe=True,          # CLAHE etkin
        enable_gamma=True,          # Gamma düzeltme etkin
        gamma_value=1.2,           # Hafif parlaklık artışı
        clahe_clip_limit=2.0,      # CLAHE clip limit
        clahe_tile_size=(8, 8)     # CLAHE tile boyutu
    )