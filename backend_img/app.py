from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from PIL import Image
import torch
from model import VIT, PatchEmbedding, multiHeadAttention, residual, mlp, TransformerBlock, Transformer, Classification
import io
import numpy as np
from image_processor import ImageProcessor
from model_loader import models
import sys
import pathlib

def get_my_x(o):
    return o

def get_my_y(o):
    return o

sys.modules['__main__'].get_my_x = get_my_x
sys.modules['__main__'].get_my_y = get_my_y

class PosixPath(str):
    def __new__(cls, *args, **kwargs):
        return str.__new__(cls, *args)

sys.modules['pathlib'].PosixPath = PosixPath

app = Flask(__name__)
CORS(app)

image_processor = ImageProcessor()

@app.route('/detect', methods=['POST'])
def detect_fake():
    try:
        data = request.json
        image_data = data.get('image')

        if not image_data:
            return jsonify({'error': 'Görsel verisi bulunamadı'}), 400

        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        processed_image = image_processor.prepare_image(image)
        results = {}

        for i, model in enumerate(models[3:]):
            prediction = image_processor.predict_pytorch(model, processed_image)
            model_name = f"model_{i+4}.pth"
            results[model_name] = {
                'fake_probability': float(prediction),
                'is_fake': prediction > 0.5
            }

        for i, model in enumerate(models[0:3]):
            features = image_processor.extract_features(processed_image)
            prediction = image_processor.predict_sklearn(model, features)
            model_name = f"model_{i+1}.pkl"
            results[model_name] = {
                'fake_probability': float(prediction),
                'is_fake': prediction > 0.5
            }

        fake_count = sum(1 for r in results.values() if r['is_fake'])
        total_models = len(results)
        overall_confidence = sum(r['fake_probability'] for r in results.values()) / total_models if total_models > 0 else 0

        final_result = {
            'is_fake': fake_count > total_models / 2,
            'confidence': overall_confidence,
            'fake_model_count': fake_count,
            'total_models': total_models,
            'individual_results': results,
            'summary': f"{fake_count}/{total_models} model sahte olarak tespit etti"
        }

        return jsonify(final_result)

    except Exception as e:
        return jsonify({'error': f'Bir hata oluştu: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': len(models)})

if __name__ == '__main__':
    print("Flask server başlatılıyor...")
    print(f"Yüklenen modeller: {len(models)}")
    app.run(debug=True, host='0.0.0.0', port=5000)
