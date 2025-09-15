import os
import pydicom
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.console import Console
import warnings

def get_window_values(dicom_image, pixel_array):
    try:
        wc = dicom_image.WindowCenter
        ww = dicom_image.WindowWidth
        wc = float(wc[0]) if isinstance(wc, pydicom.multival.MultiValue) else float(wc)
        ww = float(ww[0]) if isinstance(ww, pydicom.multival.MultiValue) else float(ww)
    except Exception:
        wc = np.mean(pixel_array)
        ww = np.max(pixel_array) - np.min(pixel_array)
    return wc, ww

def process_single_dicom(dicom_path, output_path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dicom_image = pydicom.dcmread(dicom_path, force=True)

        if not hasattr(dicom_image, "PixelData"):
            raise ValueError("PixelData etiketi bulunamadı.")

        pixel_array = dicom_image.pixel_array.astype(np.float32)
        slope = float(getattr(dicom_image, "RescaleSlope", 1.0))
        intercept = float(getattr(dicom_image, "RescaleIntercept", 0.0))
        hu_image = pixel_array * slope + intercept

        wc, ww = get_window_values(dicom_image, hu_image)
        min_val = wc - ww / 2
        max_val = wc + ww / 2
        
        if max_val == min_val:
            max_val = min_val + 1

        img_windowed = np.clip(hu_image, min_val, max_val)
        img_norm = ((img_windowed - min_val) / (max_val - min_val)) * 255.0
        img_uint8 = img_norm.astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not cv2.imwrite(output_path, img_uint8):
             raise IOError("Dosya kaydedilemedi.")

        return True, dicom_path
    except Exception as e:
        error_message = f"{dicom_path} -> {type(e).__name__}: {e}\n"
        with open("failed_files.txt", "a", encoding='utf-8') as f:
            f.write(error_message)
        return False, dicom_path

def convert_dicom_directory(input_dir, output_dir, max_workers=8):
    console = Console()
    tasks = []
    
    console.print("[yellow]🔍 DICOM dosyaları taranıyor...[/yellow]")
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                relative_path = os.path.relpath(root, input_dir)
                
                path_parts = relative_path.split(os.sep)
                if path_parts:
                    patient_id = path_parts[0] 
                    dicom_path = os.path.join(root, file)   
                    output_patient_dir = os.path.join(output_dir, patient_id)
                    output_path = os.path.join(output_patient_dir, f"{os.path.splitext(file)[0]}.png")
                    
                    tasks.append((dicom_path, output_path))

    if not tasks:
        console.print("[red]❌ Hiç DICOM dosyası bulunamadı![/red]")
        return

    console.print(f"[cyan]✨ Toplam {len(tasks)} dosya bulundu. Dönüştürme başlıyor...[/cyan]")

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), TimeElapsedColumn(), console=console
    ) as progress:
        task = progress.add_task("[green]Dönüştürülüyor...", total=len(tasks))
        success_count = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(process_single_dicom, *task_args): task_args for task_args in tasks}
            for future in as_completed(future_to_task):
                is_success, _ = future.result()
                if is_success:
                    success_count += 1
                progress.update(task, advance=1)

    console.print(f"\n[bold green]✅ İşlem tamamlandı![/bold green]")
    console.print(f"Toplam {len(tasks)} dosyadan {success_count} tanesi başarıyla dönüştürüldü.")
    failure_count = len(tasks) - success_count
    if failure_count > 0:
        console.print(f"[bold yellow]⚠️ {failure_count} dosya işlenemedi. Detaylar için 'failed_files.txt' dosyasını kontrol edin.[/bold yellow]")

if __name__ == '__main__':
    input_directory = "/home/comp5/ARTEK/SYZ_25_YARISMA/IKINCI_GOREV/MR_testset"  
    output_directory = "/home/comp5/ARTEK/SYZ_25_YARISMA/IKINCI_GOREV/MR_testset_PNG"

    WORKER_COUNT = 12 

    convert_dicom_directory(input_directory, output_directory, max_workers=WORKER_COUNT)