import os
import pydicom
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.console import Console
import warnings
import time

# --- YENİ EKLENEN FONKSİYON ---
def auto_crop(img):
    """
    Görüntüdeki en büyük nesneyi (konturu) bularak etrafındaki boş siyah alanları kırpar.
    """
    # Görüntünün tek kanallı (grayscale) olduğundan emin olalım.
    # Bu script'te gelen 'img' zaten tek kanallı olacaktır.
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Gürültüyü azaltmak için hafif bir bulanıklaştırma
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Eşikleme ile nesneyi arka plandan ayır
    # Eşik değeri (10), çok koyu piksellerin arka plan olarak kabul edilmesini sağlar.
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)

    # Görüntüdeki tüm dış konturları bul
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Eğer hiç kontur bulunamazsa orijinal resmi geri döndür
    if not contours:
        return img

    # En büyük alana sahip konturu ana nesne olarak seç
    c = max(contours, key=cv2.contourArea)

    # En büyük konturun sınırlayıcı kutusunu (bounding box) bul
    x, y, w, h = cv2.boundingRect(c)

    # Kenarlara çok yakın kırpmamak için küçük bir pay (margin) bırakalım
    margin_x = int(w * 0.04) # Genişliğin %4'ü
    margin_y = int(h * 0.05) # Yüksekliğin %5'i

    x_start = max(0, x - margin_x)
    y_start = max(0, y - margin_y)
    x_end = min(img.shape[1], x + w + margin_x)
    y_end = min(img.shape[0], y + h + margin_y)

    # Kırpılmış görüntüyü oluştur
    cropped = img[y_start:y_end, x_start:x_end]

    # Eğer kırpma sonucu çok küçük bir alan kalırsa veya boş ise orijinali kullan
    if cropped.size == 0 or cropped.shape[0] < 100 or cropped.shape[1] < 100:
        return img

    return cropped
# --- BİTİŞ ---

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

def process_batch(batch_files):
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

            # --- OTOMATİK KIRPMA ADIMI ---
            # Görüntüyü kaydetmeden hemen önce kırpıyoruz.
            img_cropped = auto_crop(img_uint8)
            # --- BİTİŞ ---

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Kırpılmış görüntüyü PNG olarak kaydet
            saved = cv2.imwrite(output_path, img_cropped)
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

def convert_dicom_directory(input_directory, output_directory, max_workers=6, batch_size=256):
    console = Console()
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

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

    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Dönüştürülüyor...", total=total_files)
        completed = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch): batch
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
    console.print("---" * 10)
    console.print(f"Toplam dönüştürülen resim sayısı: {total_files}")
    console.print(f"Toplam geçen süre: {total_time:.2f} saniye")
    console.print(f"Bir resmi ortalama dönüştürme süresi: {average_time:.4f} saniye")
    console.print(f"Paralel işlem sayısı (workers): {max_workers}")
    console.print("---" * 10)
    console.print("[yellow]Başarısız dosyalar (varsa) 'failed_files.txt' içine kaydedildi.[/yellow]")

if __name__ == '__main__':
    input_dir = "/home/comp5/ARTEK/SYZ_25_YARISMA/BIRINCI_GOREV/ornek_veriler/deneme_dcom"
    output_dir = "/home/comp5/ARTEK/SYZ_25_YARISMA/BIRINCI_GOREV/ornek_veriler/deneme_png"

    convert_dicom_directory(input_dir, output_dir, max_workers=16, batch_size=128)