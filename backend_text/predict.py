from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import sys
import os
import torch
import requests
from bs4 import BeautifulSoup
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# CORS desteği için fact_checker'ı import et
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'explanation')))
try:
    from fact_checker import get_explanation
except ImportError:
    print("Warning: fact_checker could not be imported. Using fallback explanation.")
    def get_explanation(text):
        return {
            "explanation": "Yapay zeka analizi tamamlandı. Detaylı açıklama şu anda mevcut değil.",
            "source": "Yapay Zeka Analizi"
        }

app = Flask(__name__)
CORS(app)  # Tüm rotalar için CORS'u etkinleştir

# Model ve tokenizer yükleme
try:
    model = DistilBertForSequenceClassification.from_pretrained("distilBert_model/saved_model")
    tokenizer = DistilBertTokenizer.from_pretrained("distilBert_model/saved_model")
    model.eval()
    print("✅ Model başarıyla yüklendi!")
except Exception as e:
    print(f"❌ Model yüklenirken hata: {e}")
    print("Lütfen model dosyalarının 'distilBert_model/saved_model' klasöründe olduğundan emin olun.")
    model = None
    tokenizer = None

def clean_html(url):
    """URL'den metin çıkarma fonksiyonu"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        res = requests.get(url, timeout=10, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        
        # Script ve style etiketlerini kaldır
        for script in soup(["script", "style"]):
            script.decompose()
            
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        return text.strip()
    except Exception as e:
        print(f"URL çıkarma hatası: {e}")
        return None

def predict_text(text_or_url, is_url=False):
    """Metin tahmin fonksiyonu"""
    if model is None or tokenizer is None:
        return False, 0.5, "Model yüklenemedi", "Sistem Hatası"
    
    # URL ise metni çıkar
    if is_url:
        text = clean_html(text_or_url)
        if not text:
            return False, 0.5, "URL'den metin alınamadı", "URL Hatası"
    else:
        text = text_or_url
    
    if not text or len(text.strip()) < 10:
        return False, 0.5, "Metin çok kısa veya boş", "Giriş Hatası"

    try:
        # Tokenization ve tahmin
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            pred = torch.argmax(probs, dim=1)

        is_fake = pred.item() == 1
        confidence = probs[0][1].item()
        
        # Açıklama al
        try:
            explanation_data = get_explanation(text)
            explanation = explanation_data.get("explanation", "Analiz tamamlandı")
            source = explanation_data.get("source", "AI Analysis")
        except:
            explanation = "Yapay zeka modeli analizi tamamladı"
            source = "Yapay Zeka Modeli"
        
        return is_fake, confidence, explanation, source
    
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return False, 0.5, f"Analiz sırasında hata oluştu: {str(e)}", "Analiz Hatası"

@app.route("/", methods=["GET"])
def health_check():
    """Sunucu sağlık kontrolü"""
    return jsonify({
        "status": "Flask sunucusu çalışıyor",
        "model_loaded": model is not None,
        "endpoint": "/predict"
    })

@app.route("/predict", methods=["POST", "OPTIONS"])
@cross_origin()
def api_predict():
    """Ana tahmin API'si"""
    # OPTIONS isteği için
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight OK"}), 200
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON verisi alınamadı"}), 400
            
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Boş metin gönderildi"}), 400

        print(f"📝 Analiz ediliyor: {text[:100]}...")
        
        # URL kontrolü
        is_url = text.startswith(('http://', 'https://'))
        
        is_fake, confidence, explanation, source = predict_text(text, is_url)
        
        result = {
            "is_fake": is_fake,
            "confidence": confidence,
            "message": explanation,
            "source": source,
            "input_type": "URL" if is_url else "Metin"
        }
        
        print(f"✅ Sonuç: {'SAHTE' if is_fake else 'GERÇEK'} (Güven: {confidence:.2f})")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ API Hatası: {e}")
        return jsonify({
            "error": f"Sunucu hatası: {str(e)}",
            "is_fake": False,
            "confidence": 0.5,
            "message": "Analiz sırasında beklenmeyen bir hata oluştu",
            "source": "Hata İşleyicisi"
        }), 500

if __name__ == "__main__":
    print("🚀 Flask sunucusu başlatılıyor...")
    print("📡 URL: http://localhost:5001")
    print("🔧 Endpoint: http://localhost:5001/predict")
    app.run(host="0.0.0.0", port=5001, debug=True)