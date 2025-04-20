import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from datasets import load_dataset
import pickle
import json
import numpy as np

# Download NLTK resources
try:
    nltk.download("punkt")
    nltk.download("punkt_tab")
except Exception as e:
    print(f"Error downloading NLTK resources: {e}")
    raise SystemExit("Please check internet connection and rerun.")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad_indices(indices, max_len=20):
    indices = indices[:max_len]
    indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices

def label_intent(question):
    question = clean_text(question)
    # Load keywords from JSON
    try:
        with open("data/keywords.json", "r") as f:
            keywords = json.load(f)
        symptom_keywords = keywords["symptom"]
        treatment_keywords = keywords["treatment"]
    except FileNotFoundError:
        print("Error: keywords.json not found in data/ directory")
        raise SystemExit("Please create data/keywords.json with symptom and treatment keywords")
    
    # Check for keyword presence in words or as substrings
    words = question.split()
    for keyword in symptom_keywords:
        if keyword in words or any(keyword in word for word in words if len(word) >= len(keyword)):
            # print(f"Debug: Matched symptom keyword '{keyword}' in '{question}'")
            return 0  # Symptom
    for keyword in treatment_keywords:
        if keyword in words or any(keyword in word for word in words if len(word) >= len(keyword)):
            # print(f"Debug: Matched treatment keyword '{keyword}' in '{question}'")
            return 1  # Treatment
    # print(f"Debug: No keyword matched in '{question}', defaulting to general")
    return 2  # General

# Load dataset
try:
    dataset = load_dataset("DSWF/medical_chatbot")
    data = dataset["train"].to_pandas()
    print("Dataset loaded successfully!")
    print(data.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise SystemExit("Please check internet connection or download medical_chatbot.csv manually")

data = data.dropna()
print(f"Number of rows after dropping NaN: {len(data)}")

data["Question"] = data["Question"].apply(clean_text)
data["Answer"] = data["Answer"].apply(clean_text)
data["patient_tokens"] = data["Question"].apply(word_tokenize)
data["doctor_tokens"] = data["Answer"].apply(word_tokenize)

# Load doctors.json for all_symptoms
try:
    with open("data/doctors.json", "r") as f:
        doc_data = json.load(f)
    all_symptoms = doc_data.get("all_symptoms", list(doc_data["mappings"].keys()))
except FileNotFoundError:
    print("Error: doctors.json not found in data/ directory")
    raise SystemExit("Please create data/doctors.json with all_symptoms and mappings")

# Extract symptom co-occurrence
symptom_cooc = {}
for sym1 in all_symptoms:
    symptom_cooc[sym1] = {}
    for sym2 in all_symptoms:
        symptom_cooc[sym1][sym2] = 0

for question in data["Question"]:
    words = question.split()
    present_symptoms = [sym for sym in all_symptoms if sym in words]
    for i, sym1 in enumerate(present_symptoms):
        for sym2 in present_symptoms[i+1:]:
            symptom_cooc[sym1][sym2] += 1
            symptom_cooc[sym2][sym1] += 1

# Convert to matrix
cooc_matrix = np.zeros((len(all_symptoms), len(all_symptoms)))
for i, sym1 in enumerate(all_symptoms):
    for j, sym2 in enumerate(all_symptoms):
        cooc_matrix[i][j] = symptom_cooc[sym1][sym2]

# Save co-occurrence matrix
np.save("data/symptom_cooc.npy", cooc_matrix)
print("Symptom co-occurrence matrix saved to data/symptom_cooc.npy")

all_words = [word for sent in data["patient_tokens"] for word in sent] + \
            [word for sent in data["doctor_tokens"] for word in sent]
word_counts = Counter(all_words)
vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.most_common(10000))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

data["patient_indices"] = data["patient_tokens"].apply(lambda x: tokens_to_indices(x, vocab))
data["doctor_indices"] = data["doctor_tokens"].apply(lambda x: tokens_to_indices(x, vocab))
data["patient_padded"] = data["patient_indices"].apply(lambda x: pad_indices(x))
data["doctor_padded"] = data["doctor_indices"].apply(lambda x: pad_indices(x))

data["intent"] = data["Question"].apply(label_intent)
intent_map = {0: "symptom", 1: "treatment", 2: "general"}

data[["patient_padded", "doctor_padded", "intent"]].to_pickle("data/preprocessed_data.pkl")
with open("data/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("data/intent_map.pkl", "wb") as f:
    pickle.dump(intent_map, f)

print("Preprocessing done! Saved to data/preprocessed_data.pkl")