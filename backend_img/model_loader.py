import torch
import pickle
import sys
import os
import pathlib

# FastAI ve sklearn model çözüm için gerekli fonksiyonlar
def get_my_x(o):
    return o

def get_my_y(o):
    return o

# Özel fonksiyonları __main__'e ekle
sys.modules['__main__'].get_my_x = get_my_x
sys.modules['__main__'].get_my_y = get_my_y

# PosixPath override
class PosixPath(pathlib.WindowsPath):
    _flavour = pathlib._windows_flavour

sys.modules['__main__'].PosixPath = PosixPath

# Modelden kullanılan sınıflar
from model import PatchEmbedding, Transformer, VIT, TransformerBlock, residual, multiHeadAttention

# PyTorch 1.8.1 için '__main__' atamaları
sys.modules['__main__'].PatchEmbedding = PatchEmbedding
sys.modules['__main__'].Transformer = Transformer
sys.modules['__main__'].VIT = VIT
sys.modules['__main__'].TransformerBlock = TransformerBlock
sys.modules['__main__'].residual = residual
sys.modules['__main__'].multiHeadAttention = multiHeadAttention

# Sklearn modeli yükleyici
def load_sklearn_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print(f"[✓] Sklearn modeli yüklendi: {model_path}")
        return model
    except Exception as e:
        print(f"[X] Sklearn model hatası: {model_path} → {e}")
        return None

# Torch modeli yükleyici
def load_torch_model(model_path):
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        print(f"[✓] Torch modeli yüklendi: {model_path}")
        return model
    except Exception as e:
        print(f"[X] Torch model hatası: {model_path} → {e}")
        return None

# Model yolları
model_paths = [
    "models/Copy_Move_FIM_fixed.pkl",
    "models/Inpainting_FIM_fixed.pkl",
    "models/Splicing_FIM_fixed.pkl",
    "models/Deepfake_Model_1.pth",
    "models/Deepfake_Model_2.pth",
    "models/Fake_Face_Detection_Model.pth",
    "models/Fake_Face_Detection_Model_1.pth",
    "models/Fake_Face_Detection_Model_2.pth",
    "models/model_2.pth"
]

# Modelleri yükle
models = []
for path in model_paths:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pkl":
        models.append(load_sklearn_model(path))
    elif ext == ".pth":
        models.append(load_torch_model(path))