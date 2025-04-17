import torch
import nltk
from language_tool_python import LanguageTool
import pickle
import json
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
    symptom_counts = {}
    for symptom in symptoms:
        if symptom in doctor_mappings["mappings"]:
            for doctor in doctor_mappings["mappings"][symptom]:
                symptom_counts[doctor] = symptom_counts.get(doctor, 0) + 1
    if not symptom_counts:
        return doctor_mappings["default"][0]
    return max(symptom_counts.items(), key=lambda x: x[1])[0]

def check_negation(text, tool):
    matches = tool.check(text)
    for match in matches:
        if "negation" in match.category or any(neg in match.context.lower() for neg in ["not", "no", "without", "n't"]):
            return True
    return False

def chat(model, vocab, device, intent_map, tool, max_len=20):
    model.eval()
    predefined_responses = {
        0: "Please describe your symptoms in detail.",  # Symptom (default)
        1: "I can suggest general treatments, but consult a doctor for prescriptions.",  # Treatment
        2: "Can you clarify your question or provide more details?"  # General
    }
    try:
        with open("data/keywords.json", "r") as f:
            keywords = json.load(f)
        symptom_keywords = keywords["symptom"]
        treatment_keywords = keywords["treatment"]
        with open("data/doctors.json", "r") as f:
            doctor_mappings = json.load(f)
    except FileNotFoundError:
        print("Error: keywords.json or doctors.json not found in data/ directory")
        raise SystemExit("Please create data/keywords.json and data/doctors.json with symptom and doctor mappings")
    
    base_responses = {
        "fever": "Fever could indicate an infection. Any other symptoms like {related}?",
        "cough": "A cough might suggest a cold or respiratory issue. Is it dry or productive? Any other symptoms like {related}?",
        "headache": "Headaches can have many causes. Are you experiencing stress, dehydration, or other symptoms like {related}?",
        "chills": "Chills often accompany fever. Are you experiencing sweating or body aches? Any other symptoms like {related}?",
        "cold": "A cold might include symptoms like {related}. Any of these?",
        "<default>": "Please describe this symptom in detail or consult a doctor for advice. Any related symptoms like {related}?"
    }
    
    conversation_history = []
    last_response = ""
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
        
        is_vague = cleaned_input in ["yeah", "yes", "no", "no other symptoms", "all symptoms are provided"]
        if is_vague and conversation_history:
            if cleaned_input in ["yeah", "yes"]:
                last_entry = conversation_history[-1]
                detected_symptoms = last_entry["symptoms"]
                if detected_symptoms:
                    response = f"Okay, you confirmed {', '.join(detected_symptoms)}. Any other symptoms?"
                    past_symptoms = []
                    for past_entry in conversation_history[:-1]:
                        past_symptoms.extend(past_entry["symptoms"])
                    past_symptoms = list(set(past_symptoms) - set(detected_symptoms))
                    if past_symptoms:
                        response += f" You previously mentioned {', '.join(past_symptoms)}. Please provide more details."
                    intent = 0
                else:
                    response = "Can you specify which symptoms you're confirming?"
                    intent = 2
            elif cleaned_input in ["no", "no other symptoms"]:
                all_symptoms = []
                for entry in conversation_history:
                    all_symptoms.extend(entry["symptoms"])
                all_symptoms = list(set(all_symptoms))
                if all_symptoms:
                    response = f"You mentioned {', '.join(all_symptoms)}. No other symptoms? If you have mentioned all the symptoms then please type all symptoms are provided"
                    intent = 0
                else:
                    response = "You haven't mentioned any symptoms yet. Please describe how you're feeling."
                    intent = 2
            elif cleaned_input == "all symptoms are provided":
                all_symptoms = []
                for entry in conversation_history:
                    all_symptoms.extend(entry["symptoms"])
                all_symptoms = list(set(all_symptoms))
                if all_symptoms:
                    doctor_type = recommend_doctor(all_symptoms, doctor_mappings)
                    response = f"You mentioned {', '.join(all_symptoms)}. Based on your symptoms, I recommend consulting a {doctor_type}."
                    intent = 0
                else:
                    response = "You haven't mentioned any symptoms yet. Please describe how you're feeling."
                    intent = 2
            conversation_history.append({"input": cleaned_input, "symptoms": []})
            print(f"Debug: Cleaned input: '{cleaned_input}', Predicted intent: {intent} ({intent_map[intent]})")
            print(f"Doctor ({intent_map[intent]}): {response}")
            last_response = response
            continue
        
        detected_symptoms = []
        is_negated = check_negation(cleaned_input, tool)
        if not is_negated and any(keyword in words for keyword in symptom_keywords):
            for i, word in enumerate(words):
                if word in symptom_keywords:
                    detected_symptoms.append(word)
        
        tokens = cleaned_input.split()  # Simplified tokenization
        input_indices = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
        input_indices = pad_indices(input_indices, max_len)
        src = torch.tensor([input_indices], dtype=torch.long).to(device)
        hidden, cell = model.encoder(src)
        intent_logits = model.classifier(hidden[-1])
        intent = intent_logits.argmax(1).item()
        
        print(f"Debug: Cleaned input: '{cleaned_input}', Predicted intent: {intent} ({intent_map[intent]})")
        
        if any(keyword in words for keyword in symptom_keywords):
            print(f"Debug: Overriding intent to symptom due to keyword match in '{cleaned_input}'")
            intent = 0
        elif any(phrase in cleaned_input for phrase in ["what are the other symptom", "tell me the symptoms", "other symptoms for", "i am feeling cold"]):
            print(f"Debug: Overriding intent to symptom or list request in '{cleaned_input}'")
            intent = 0
        elif any(keyword in words for keyword in treatment_keywords):
            print(f"Debug: Overriding intent to treatment due to keyword match in '{cleaned_input}'")
            intent = 1
        
        conversation_history.append({"input": cleaned_input, "symptoms": detected_symptoms})
        
        if any(phrase in cleaned_input for phrase in ["what are the other symptom", "tell me the symptoms", "other symptoms for", "i am feeling cold"]):
            if "cold" in detected_symptoms or "i am feeling cold" in cleaned_input:
                all_past_symptoms = set()
                for entry in conversation_history:
                    all_past_symptoms.update(entry["symptoms"])
                related_symptoms = ", ".join([kw for kw in symptom_keywords if kw not in all_past_symptoms][:5])
                response = base_responses["cold"].format(related=related_symptoms)
            else:
                related_symptoms = ", ".join(symptom_keywords[:5])
                response = f"Common symptoms I can recognize include {related_symptoms} and more. Please specify a condition for related symptoms."
        else:
            if intent == 0:  # Symptom
                if detected_symptoms:
                    past_symptoms = []
                    for past_entry in conversation_history[:-1]:
                        past_symptoms.extend(past_entry["symptoms"])
                    new_symptoms = [s for s in detected_symptoms if s not in past_symptoms]
                    if new_symptoms:
                        all_past_symptoms = set()
                        for entry in conversation_history:
                            all_past_symptoms.update(entry["symptoms"])
                        related_symptoms = ", ".join([kw for kw in symptom_keywords if kw not in all_past_symptoms][:5])
                        response = base_responses.get(new_symptoms[0], base_responses["<default>"]).format(related=related_symptoms)
                    else:
                        all_past_symptoms = set()
                        for entry in conversation_history:
                            all_past_symptoms.update(entry["symptoms"])
                        related_symptoms = ", ".join([kw for kw in symptom_keywords if kw not in all_past_symptoms][:5])
                        response = base_responses.get(detected_symptoms[0], base_responses["<default>"]).format(related=related_symptoms)
                    if len(detected_symptoms) > 1:
                        response += f" You also mentioned {', '.join(detected_symptoms[1:])} in this input."
                    past_symptoms = list(set(past_symptoms))
                    if past_symptoms:
                        response += f" You previously mentioned {', '.join(past_symptoms)}. Please provide more details about all symptoms."
                else:
                    response = predefined_responses[0]
            else:
                response = predefined_responses.get(intent, "I'm not sure how to respond.")
        
        print(f"Doctor ({intent_map[intent]}): {response}")
        last_response = response

chat(model, vocab, device, intent_map, tool)