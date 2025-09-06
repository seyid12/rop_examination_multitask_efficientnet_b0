# ROP Multitask Prediction API

This project provides a prediction API based on **FastAPI** for retinal optic plaque (ROP) diagnosis and pathology classification. The model is trained with **PyTorch** and based on **EfficientNet-B0**.

---

## Features

- Predicts **ROP degree (DG)** and **Pathology class (PF)** by uploading images.
- Returns results in JSON format.
- Can run on GPU (CUDA) or CPU.

---

## Server startup

```bash
uvicorn app:app --reload
```

---

## Kurulum

1. Clone the repository:

```bash
git clone https://github.com/kullanici_adi/rop_multitask_api.git
cd rop_multitask_api
```
## Server startup

```bash
uvicorn app:app --reload
```
## Once the server is running:

```cpp
http://127.0.0.1:8000
```
You can now access the API.

## Test script

You can use test_request.py to send an image and get predictions:
```python
import requests

url = "http://127.0.0.1:8000/predict"
image_path = "gorsel.jpg"  # test görselin

with open(image_path, "rb") as f:
    files = {"file": (image_path.split("/")[-1], f, "image/jpeg")}
    response = requests.post(url, files=files)
    print(response.json())
```
Example output:
```json
{
    "dg_pred": 1,
    "pf_pred": 0,
    "pf_prob": [0.7, 0.3, 0.0]
}
```

## API Endpoint
Endpoint Method Description
/predict POST Uploads a single image and gets DG and PF predictions.

Request:

- Upload an image to the file field in form-data.

Response:

- dg_pred → ROP grade prediction

- pf_pred → Pathology class prediction

- pf_prob → PF class probabilities (list)

Model Information

- Backbone: EfficientNet-B0

- Multitask Head:

- - dg_head: ROP grade prediction

- - pf_head: Pathology class prediction

- ImageNet pre-weights were used during training.

- Model checkpoint file: rop_multitask_efficientnet_b0.pt

## Requirements

- Python 3.10+

- FastAPI

- Uvicorn

- Torch

- Torchvision

- Pillow

- Requests (for testing)
