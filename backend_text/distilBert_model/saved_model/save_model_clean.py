from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# TAM yol (absolute path) dikkat: ters slash yerine çift ters slash!
checkpoint_path = "C:/Users/HpVictus/Desktop/fakeNewsDetector/fakeNewsDetector/model/DistilBert/results/checkpoint-10000"

# Model ve tokenizer'ı yükle
model = DistilBertForSequenceClassification.from_pretrained(checkpoint_path)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

# Kaydedilecek klasör
save_path = "saved_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Model ve tokenizer başarıyla 'saved_model/' klasörüne kaydedildi.")

