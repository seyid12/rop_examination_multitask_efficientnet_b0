from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
from PIL import Image
from torchvision import transforms, models
import io

app = FastAPI()

# Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modeli yükle
ckpt_path = "rop_multitask_efficientnet_b0.pt"
ckpt = torch.load(ckpt_path, map_location=device)

# Checkpoint'teki gerçek sınıf sayısını al
num_dg = ckpt["model_state"]["dg_head.weight"].shape[0]
num_pf = ckpt["model_state"]["pf_head.weight"].shape[0]

class ROPNet(torch.nn.Module):
    def __init__(self, num_dg, num_pf):
        super().__init__()
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_dim = base.classifier[1].in_features
        self.backbone = base.features
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.dropout = torch.nn.Dropout(0.2)
        self.dg_head = torch.nn.Linear(feat_dim, num_dg)
        self.pf_head = torch.nn.Linear(feat_dim, num_pf)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        dg_logits = self.dg_head(x)
        pf_logits = self.pf_head(x)
        return dg_logits, pf_logits

# Modeli oluştur ve checkpoint ile yükle (strict=False)
model = ROPNet(num_dg, num_pf).to(device)
model.load_state_dict(ckpt["model_state"], strict=False)
model.eval()

# Normalizasyon
IM_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(int(IM_SIZE*1.15)),
    transforms.CenterCrop(IM_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            dg_logits, pf_logits = model(img)
            dg_pred = torch.argmax(dg_logits, dim=1).item()
            pf_pred = torch.argmax(pf_logits, dim=1).item()
            pf_prob = torch.softmax(pf_logits, dim=1).cpu().numpy().tolist()[0]

        return JSONResponse({
            "dg_pred": dg_pred,
            "pf_pred": pf_pred,
            "pf_prob": pf_prob
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
