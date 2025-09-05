import requests

# FastAPI server URL
url = "http://127.0.0.1:8000/predict"

# Test edilecek görsel dosya yolu
image_path = r"C:\Users\Seyid\OneDrive - ogr.dicle.edu.tr\Masaüstü\fastapi\001_F_GA41_BW2905_PA44_DG11_PF0_D1_S01_1.jpg"

# Dosyayı gönder
with open(image_path, "rb") as f:
    files = {"file": (image_path.split("\\")[-1], f, "image/jpeg")}
    try:
        response = requests.post(url, files=files)
        response.raise_for_status()  # HTTP hatalarını tetikler
        data = response.json()
        print("DG Tahmin:", data["dg_pred"])
        print("PF Tahmin:", data["pf_pred"])
        print("PF Olasılıkları:", data["pf_prob"])
    except requests.exceptions.RequestException as e:
        print("Hata:", e)
