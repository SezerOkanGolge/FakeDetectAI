import transformers
print("Transformers version:", transformers.__version__)
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    print("Using CPU. Check CUDA or driver installation.")

import sys
print("PYTHON:", sys.executable)
import pandas as pd
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

#Verileri oku
df_tr = pd.read_csv("../data/dataSet_TR_cleaned.csv")[["clean_text", "label"]]
df_en = pd.read_csv("../data/dataSet_EN_cleaned.csv")[["clean_text", "label"]]

# Birleştir ve temizle
df_all = pd.concat([df_tr, df_en], ignore_index=True)
df_all = df_all.dropna()
df_all["label"] = df_all["label"].astype(int)

# Sınıf dengesini kur (0 olanları çoğalt)
df_majority = df_all[df_all.label == 1]
df_minority = df_all[df_all.label == 0]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

# Train-test ayır
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["clean_text"], df_balanced["label"],
    test_size=0.2, stratify=df_balanced["label"], random_state=42
)

from transformers import DistilBertTokenizer
from datasets import Dataset

# 1. Tokenizer'ı yükle
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-multilingual-cased")

# 2. Tokenize etme fonksiyonu
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

# 3. Eğitim ve test veri setini Hugging Face Dataset formatına çevir
train_dataset = Dataset.from_dict({
    "text": list(X_train),
    "label": list(map(int, y_train))
})
test_dataset = Dataset.from_dict({
    "text": list(X_test),
    "label": list(map(int, y_test))
})

test_dataset = Dataset.from_dict({"text": X_test.tolist(), "label": y_test.tolist()})

# 4. Tokenization uygula
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)
#train_dataset = train_dataset.select(range(5000))  # sadece ilk 5000 veri
#test_dataset = test_dataset.select(range(1000))    # test için 1000 veri yeterli


from transformers import DistilBertForSequenceClassification

# Modeli yükle (etiket sayısı: 2 → sahte / gerçek)
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-multilingual-cased",
    num_labels=2
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    logging_dir="./logs",
    fp16=True  # GPU hızlandırma
)

from transformers import Trainer

# Doğruluk hesabı için yardımcı fonksiyon
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# Trainer nesnesini oluştur
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Modeli eğit
trainer.train()

# Modeli değerlendir (test seti üstünde)
eval_results = trainer.evaluate()
print("Değerlendirme Sonuçları:", eval_results)

eval_result = trainer.evaluate()
print("Test Doğruluğu (Accuracy):", eval_result["eval_accuracy"])
# MODELI VE TOKENIZER'I KAYDET
model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")


