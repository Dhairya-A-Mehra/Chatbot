import torch
import nltk
from language_tool_python import LanguageTool
import pickle
import json
import numpy as np
from sklearn.cluster import KMeans
import joblib
from model import Encoder, Decoder, IntentClassifier, Chatbot

# Try to use existing NLTK data, skip download if offline
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK data not found locally. Please download 'punkt' and 'punkt_tab' manually to C:\\Users\\A R Mehra\\AppData\\Roaming\\nltk_data\\tokenizers\\ and rerun.")
    exit()

# Load LanguageTool for grammar and negation checks
tool = LanguageTool('en-US')

with open("data/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("data/intent_map.pkl", "rb") as f:
    intent_map = pickle.load(f)
with open("data/doctors.json", "r") as f:
    doc_data = json.load(f)

VOCAB_SIZE = len(vocab)
EMBED_SIZE = 64
HIDDEN_SIZE = 128
NUM_CLASSES = 3
encoder = Encoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
decoder = Decoder(VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE)
classifier = IntentClassifier(HIDDEN_SIZE, 64, NUM_CLASSES)
model = Chatbot(encoder, decoder, classifier)
model.load_state_dict(torch.load("data/model.pth"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def clean_text(text):
    import re
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def pad_indices(indices, max_len=20):
    indices = indices[:max_len]
    indices += [vocab["<PAD>"]] * (max_len - len(indices))
    return indices

def recommend_doctor(symptoms, doctor_mappings):
    all_symptoms = doctor_mappings.get("all_symptoms", list(doctor_mappings["mappings"].keys()))
    cooc_matrix = np.load("data/symptom_cooc.npy")
    symptom_vec = np.zeros(len(all_symptoms))
    for sym in symptoms:
        if sym in all_symptoms:
            symptom_vec[all_symptoms.index(sym)] = 1
    
    # First check if any symptom has specific doctor mappings
    for sym in symptoms:
        if sym in doctor_mappings["mappings"]:
            return doctor_mappings["mappings"][sym][0]  # Return first recommended doctor
    
    # If no specific mapping, use clustering
    kmeans = joblib.load("data/kmeans_model.joblib")
    cluster_label = kmeans.predict([symptom_vec])[0]
    
    # Get all unique doctor types from the mappings
    all_doctor_types = set()
    for doctors in doctor_mappings["mappings"].values():
        all_doctor_types.update(doctors)
    all_doctor_types = sorted(list(all_doctor_types))
    
    # Map cluster to doctor type (using modulo to handle more clusters than doctors)
    if all_doctor_types:
        doctor_index = cluster_label % len(all_doctor_types)
        return all_doctor_types[doctor_index]
    
    return doctor_mappings["default"][0]

def check_negation(text, tool):
    matches = tool.check(text)
    for match in matches:
        if "negation" in match.category or any(neg in match.context.lower() for neg in ["not", "no", "without", "n't"]):
            return True
    return False

def get_related_symptoms(current_symptom, all_symptoms, cooc_matrix, past_symptoms):
    if current_symptom not in all_symptoms:
        return []  # Return empty if symptom not recognized
    
    sym_idx = all_symptoms.index(current_symptom)
    cooc_scores = cooc_matrix[sym_idx]
    
    # Get indices sorted by co-occurrence strength (descending), excluding current symptom
    related_indices = np.argsort(cooc_scores)[::-1]
    related_symptoms = []
    
    for idx in related_indices:
        symptom = all_symptoms[idx]
        if (symptom != current_symptom and 
            symptom not in past_symptoms and
            cooc_scores[idx] > 0):
            related_symptoms.append(symptom)
            if len(related_symptoms) >= 5:  # Limit to top 5
                break
    
    return related_symptoms

def chat(model, vocab, device, intent_map, tool, max_len=20):
    model.eval()
    predefined_responses = {
        0: "Please describe your symptoms in detail.",  # Symptom
        1: "I can suggest general treatments, but consult a doctor for prescriptions.",  # Treatment
        2: "Can you clarify your question or provide more details?"  # General
    }
    
    # Load co-occurrence matrix
    cooc_matrix = np.load("data/symptom_cooc.npy")
    all_symptoms = doc_data.get("all_symptoms", list(doc_data["mappings"].keys()))

    conversation_history = []
    print("Welcome to the Medical Chatbot! Ask about symptoms or treatments. Type 'exit' to quit.")
    print("Disclaimer: This is a demo chatbot. Consult a doctor for real medical advice.")
    
    while True:
        user_input = input("Patient: ")
        if user_input.lower() == "exit":
            break
        if not user_input.strip():
            print("Doctor: Please provide a question or symptom.")
            continue
            
        cleaned_input = clean_text(user_input)
        words = cleaned_input.split()
        
        # Handle vague responses
        is_vague = cleaned_input in ["yeah", "yes", "no", "no other symptoms", "all symptoms are provided"]
        if is_vague and conversation_history:
            last_entry = conversation_history[-1]
            detected_symptoms = last_entry["symptoms"]
            
            if cleaned_input in ["yeah", "yes"]:
                if detected_symptoms:
                    past_symptoms = []
                    for entry in conversation_history[:-1]:
                        past_symptoms.extend(entry["symptoms"])
                    past_symptoms = list(set(past_symptoms) - set(detected_symptoms))
                    
                    response = f"Okay, you confirmed {', '.join(detected_symptoms)}. "
                    if past_symptoms:
                        response += f"You previously mentioned {', '.join(past_symptoms)}. "
                    response += "Any other symptoms?"
                    intent = 0
                else:
                    response = "Can you specify which symptoms you're confirming?"
                    intent = 2
                    
            elif cleaned_input in ["no", "no other symptoms"]:
                mentioned_symptoms = []
                for entry in conversation_history:
                    mentioned_symptoms.extend(entry["symptoms"])
                mentioned_symptoms = list(set(mentioned_symptoms))
                
                if mentioned_symptoms:
                    response = f"You mentioned {', '.join(mentioned_symptoms)}. No other symptoms? If done, type 'all symptoms are provided'."
                    intent = 0
                else:
                    response = "You haven't mentioned any symptoms yet. Please describe how you're feeling."
                    intent = 2
                    
            elif cleaned_input == "all symptoms are provided":
                mentioned_symptoms = []
                for entry in conversation_history:
                    mentioned_symptoms.extend(entry["symptoms"])
                mentioned_symptoms = list(set(mentioned_symptoms))
                
                if mentioned_symptoms:
                    doctor_type = recommend_doctor(mentioned_symptoms, doc_data)
                    response = f"Based on your symptoms ({', '.join(mentioned_symptoms)}), I recommend consulting a {doctor_type}."
                    intent = 0
                else:
                    response = "You haven't mentioned any symptoms yet. Please describe how you're feeling."
                    intent = 2
            
            conversation_history.append({"input": cleaned_input, "symptoms": []})
            print(f"Doctor ({intent_map[intent]}): {response}")
            continue
        
        # Detect symptoms in input
        detected_symptoms = []
        is_negated = check_negation(cleaned_input, tool)
        
        if not is_negated:
            # Check single words
            for word in words:
                if word in all_symptoms:
                    detected_symptoms.append(word)
            
            # Check multi-word phrases
            for i in range(len(words) - 1):
                phrase = f"{words[i]} {words[i+1]}"
                if phrase in all_symptoms:
                    detected_symptoms.append(phrase)
        
        # Classify intent
        tokens = cleaned_input.split()
        input_indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        input_indices = pad_indices(input_indices, max_len)
        src = torch.tensor([input_indices], dtype=torch.long).to(device)
        hidden, cell = model.encoder(src)
        intent_logits = model.classifier(hidden[-1])
        intent = intent_logits.argmax(1).item()
        
        # Override intent if symptoms detected
        if detected_symptoms:
            intent = 0  # Symptom intent
        
        conversation_history.append({"input": cleaned_input, "symptoms": detected_symptoms})
        
        # Generate response
        if intent == 0:  # Symptom
            if detected_symptoms:
                past_symptoms = []
                for entry in conversation_history[:-1]:
                    past_symptoms.extend(entry["symptoms"])
                past_symptoms = list(set(past_symptoms))
                
                new_symptoms = [s for s in detected_symptoms if s not in past_symptoms]
                
                if new_symptoms:
                    primary_symptom = new_symptoms[0]
                    related_symptoms = get_related_symptoms(primary_symptom, all_symptoms, cooc_matrix, past_symptoms)
                    
                    if related_symptoms:
                        response = f"{primary_symptom} may be related to: {', '.join(related_symptoms)}. Are you experiencing any of these?"
                    else:
                        response = f"Tell me more about your {primary_symptom}. Any other symptoms?"
                    
                    if len(detected_symptoms) > 1:
                        response += f" (You also mentioned: {', '.join(detected_symptoms[1:])})"
                
                else:  # No new symptoms
                    primary_symptom = detected_symptoms[0]
                    related_symptoms = get_related_symptoms(primary_symptom, all_symptoms, cooc_matrix, past_symptoms)
                    
                    if related_symptoms:
                        response = f"Regarding your {primary_symptom}, are you also experiencing: {', '.join(related_symptoms)}?"
                    else:
                        response = f"Can you tell me more about your {primary_symptom}?"
            else:
                response = predefined_responses[0]
        
        else:  # Treatment or general intent
            trg = torch.zeros(1, 20).long().to(device)
            output, _ = model(src, trg, teacher_forcing_ratio=0.0)
            decoded_words = [list(vocab.keys())[list(vocab.values()).index(idx)] 
                           for idx in output.argmax(2).squeeze().tolist() 
                           if idx not in [vocab["<PAD>"], vocab["<UNK>"]]]
            response = " ".join(decoded_words[:10]) if decoded_words else predefined_responses.get(intent, "I'm not sure how to respond to that.")
        
        print(f"Doctor ({intent_map[intent]}): {response}")

if __name__ == "__main__":
    chat(model, vocab, device, intent_map, tool)