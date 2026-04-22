import spacy
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Custom symptom list (expanded from the notebook)
SYMPTOM_LIST = [
    "chest pain","back pain","headache","fever","cough","fatigue",
    "shortness of breath","nausea","vomiting","dizziness","diarrhea",
    "constipation","abdominal pain","sore throat","runny nose",
    "muscle pain","joint pain","weight loss","weight gain",
    "loss of appetite","night sweats","chills","rash","itching",
    "blurred vision","double vision","ear pain","hearing loss",
    "palpitations","swelling","anxiety","depression","insomnia",
    "confusion","memory loss","fainting","seizures","dry mouth",
    "excessive thirst","frequent urination","burning urination",
    "blood in urine","yellow skin","hair loss","brittle nails",
    "cold intolerance","heat intolerance","snoring","wheezing",
    "chest tightness","leg pain","arm pain","neck pain",
    "shoulder pain","hip pain","knee pain","stomach bloating"
]

# Training data for Logistic Regression
data = [
    ("chest pain shortness of breath sweating nausea", "heart attack"),
    ("fever cough fatigue sore throat", "flu"),
    ("headache nausea dizziness sensitivity to light", "migraine"),
    ("high blood sugar frequent urination excessive thirst", "diabetes"),
    ("weight loss heat intolerance anxiety palpitations", "hyperthyroidism"),
    ("weight gain cold intolerance fatigue depression", "hypothyroidism"),
    ("chest tightness wheezing shortness of breath", "asthma"),
    ("fever rash joint pain fatigue", "dengue"),
    ("fever chills sweating headache", "malaria"),
    ("abdominal pain diarrhea vomiting fever", "food poisoning"),
    ("persistent cough weight loss night sweats", "tuberculosis"),
    ("runny nose sneezing sore throat cough", "common cold"),
    ("joint pain stiffness swelling", "arthritis"),
    ("yellow skin abdominal pain fatigue", "liver disease"),
    ("burning urination frequent urination fever", "urinary tract infection"),
    ("memory loss confusion difficulty thinking", "alzheimer"),
    ("sadness fatigue loss of interest insomnia", "depression"),
    ("anxiety restlessness rapid heartbeat", "anxiety disorder"),
    ("fever headache stiff neck sensitivity to light", "meningitis"),
    ("leg swelling pain redness", "deep vein thrombosis"),
    ("chest pain arm pain sweating nausea", "angina"),
    ("dry cough fatigue shortness of breath", "covid-19"),
    ("itching rash redness swelling", "allergy"),
    ("ear pain hearing loss dizziness", "ear infection"),
    ("blurred vision eye pain headache", "glaucoma"),
    ("back pain stiffness reduced mobility", "spinal disorder"),
    ("abdominal pain bloating constipation", "irritable bowel syndrome"),
    ("vomiting dehydration weakness", "gastroenteritis"),
    ("snoring daytime sleepiness fatigue", "sleep apnea"),
    ("hair loss fatigue weight changes", "hormonal imbalance")
]

# Training data for PyTorch GRU model
gru_data = [
    "<START> heart attack is a serious heart condition caused by blockage of blood flow to heart muscles symptoms include chest pain shortness of breath sweating nausea treatment requires immediate medical attention <END>",
    "<START> angina is a heart condition caused by reduced blood flow to the heart symptoms include chest pain arm pain discomfort treatment includes medication and lifestyle changes <END>",
    "<START> flu is a viral infection affecting respiratory system symptoms include fever cough sore throat fatigue treatment includes rest fluids and medication <END>",
    "<START> common cold is a mild viral infection symptoms include runny nose sneezing sore throat cough treatment includes rest and hydration <END>",
    "<START> covid-19 is a viral disease caused by coronavirus symptoms include fever dry cough breathing difficulty fatigue treatment includes isolation and supportive care <END>",
    "<START> migraine is a neurological disorder symptoms include severe headache nausea dizziness sensitivity to light treatment includes pain relief and rest <END>",
    "<START> tension headache is a common condition symptoms include mild to moderate head pain pressure around head treatment includes rest and stress management <END>",
    "<START> diabetes is a chronic disease affecting blood sugar levels symptoms include excessive thirst frequent urination fatigue treatment includes diet control medication and insulin <END>",
    "<START> hypertension is a condition of high blood pressure symptoms include headache dizziness chest discomfort treatment includes lifestyle changes and medication <END>",
    "<START> hypotension is a condition of low blood pressure symptoms include dizziness fainting blurred vision treatment includes fluids and medical care <END>",
    "<START> asthma is a respiratory disease causing airway inflammation symptoms include wheezing shortness of breath chest tightness treatment includes inhalers and avoiding triggers <END>",
    "<START> bronchitis is a respiratory condition symptoms include cough mucus fatigue chest discomfort treatment includes rest fluids and medication <END>",
    "<START> pneumonia is a lung infection symptoms include fever cough breathing difficulty chest pain treatment includes antibiotics and hospitalization if severe <END>",
    "<START> tuberculosis is a bacterial infection affecting lungs symptoms include persistent cough weight loss night sweats treatment includes long term antibiotics <END>",
    "<START> malaria is a mosquito borne disease symptoms include fever chills sweating headache treatment includes antimalarial drugs <END>",
    "<START> dengue is a viral infection spread by mosquitoes symptoms include fever rash joint pain headache treatment includes hydration and medical care <END>",
    "<START> typhoid is a bacterial infection symptoms include prolonged fever abdominal pain weakness treatment includes antibiotics <END>",
    "<START> food poisoning is caused by contaminated food symptoms include vomiting diarrhea abdominal pain treatment includes hydration and rest <END>",
    "<START> gastroenteritis is an intestinal infection symptoms include diarrhea vomiting dehydration treatment includes fluids and rest <END>",
    "<START> irritable bowel syndrome is a digestive disorder symptoms include abdominal pain bloating constipation treatment includes diet changes and stress management <END>",
    "<START> liver disease is a condition affecting liver function symptoms include yellow skin abdominal pain fatigue treatment depends on underlying cause <END>",
    "<START> kidney stones is a condition caused by mineral deposits symptoms include severe abdominal pain blood in urine nausea treatment includes hydration and medication <END>",
    "<START> urinary tract infection is a bacterial infection symptoms include burning urination frequent urination fever treatment includes antibiotics <END>",
    "<START> arthritis is a joint disorder symptoms include joint pain stiffness swelling treatment includes medication and physiotherapy <END>",
    "<START> osteoporosis is a bone disease symptoms include weak bones fractures back pain treatment includes calcium supplements and medication <END>",
    "<START> depression is a mental health disorder symptoms include sadness fatigue loss of interest insomnia treatment includes therapy and medication <END>",
    "<START> anxiety disorder is a mental condition symptoms include anxiety restlessness rapid heartbeat treatment includes therapy and relaxation techniques <END>",
    "<START> insomnia is a sleep disorder symptoms include difficulty sleeping fatigue irritability treatment includes sleep hygiene and medication <END>",
    "<START> sleep apnea is a sleep disorder symptoms include snoring daytime sleepiness fatigue treatment includes breathing devices and lifestyle changes <END>",
    "<START> meningitis is an infection of brain membranes symptoms include fever headache stiff neck sensitivity to light treatment requires urgent medical care <END>",
    "<START> epilepsy is a neurological disorder symptoms include seizures confusion loss of awareness treatment includes medication <END>",
    "<START> alzheimer disease is a neurological disorder symptoms include memory loss confusion difficulty thinking treatment includes supportive care <END>",
    "<START> parkinson disease is a neurological disorder symptoms include tremors stiffness slow movement treatment includes medication and therapy <END>",
    "<START> glaucoma is an eye disease symptoms include eye pain blurred vision headache treatment includes medication or surgery <END>",
    "<START> cataract is an eye condition symptoms include blurred vision difficulty seeing at night treatment includes surgery <END>",
    "<START> ear infection is a condition affecting ear symptoms include ear pain hearing loss dizziness treatment includes medication <END>",
    "<START> sinusitis is a sinus infection symptoms include facial pain headache nasal congestion treatment includes medication and steam inhalation <END>",
    "<START> allergy is an immune system reaction symptoms include itching rash swelling sneezing treatment includes antihistamines <END>",
    "<START> eczema is a skin condition symptoms include dry skin itching redness treatment includes moisturizers and medication <END>",
    "<START> psoriasis is a skin disease symptoms include red patches itching scaling treatment includes topical therapy and medication <END>",
    "<START> anemia is a blood disorder symptoms include fatigue weakness pale skin treatment includes iron supplements <END>",
    "<START> dehydration is a condition caused by lack of fluids symptoms include thirst dizziness weakness treatment includes fluid intake <END>",
    "<START> obesity is a condition of excess body fat symptoms include weight gain fatigue treatment includes diet and exercise <END>",
    "<START> hypothyroidism is a hormonal disorder symptoms include weight gain fatigue cold intolerance treatment includes hormone therapy <END>",
    "<START> hyperthyroidism is a hormonal disorder symptoms include weight loss anxiety heat intolerance treatment includes medication <END>",
    "<START> deep vein thrombosis is a blood clot condition symptoms include leg pain swelling redness treatment includes blood thinners <END>",
    "<START> angioedema is a swelling condition symptoms include swelling of face lips throat treatment includes antihistamines and emergency care <END>",
    "<START> fibromyalgia is a chronic condition symptoms include widespread pain fatigue sleep problems treatment includes medication and therapy <END>",
    "<START> chronic fatigue syndrome is a disorder symptoms include extreme fatigue weakness treatment includes lifestyle management <END>",
    "<START> lupus is an autoimmune disease symptoms include joint pain fatigue rash treatment includes medication <END>",
    "<START> multiple sclerosis is a neurological disease symptoms include weakness vision problems coordination issues treatment includes medication <END>",
    "<START> hepatitis is a liver infection symptoms include yellow skin fatigue abdominal pain treatment includes antiviral medication <END>",
    "<START> bronchial infection is a respiratory infection symptoms include cough mucus chest discomfort treatment includes rest and medication <END>",
    "<START> throat infection is a condition affecting throat symptoms include sore throat pain swallowing difficulty treatment includes medication <END>",
    "<START> viral fever is a common infection symptoms include fever body ache fatigue treatment includes rest and fluids <END>",
    "<START> bacterial infection is a condition caused by bacteria symptoms include fever swelling pain treatment includes antibiotics <END>",
    "<START> fungal infection is a skin condition symptoms include itching redness irritation treatment includes antifungal medication <END>"
]

# --------------------------------------------------------------------------
# Models and Helpers
# --------------------------------------------------------------------------
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = self.fc(out)
        return out, hidden

def tokenize(text):
    return text.lower().split()

def build_vocab(texts):
    words = set()
    for t in texts:
        words.update(tokenize(t))
    vocab = {w: i for i, w in enumerate(words)}
    return vocab

def build_kb(data_list):
    kb = {}
    for text in data_list:
        clean = text.replace("<START>", "").replace("<END>", "").strip()
        disease = clean.split(" is ")[0]
        kb[disease] = clean
    return kb

@st.cache_resource
def load_models():
    """
    Initializes and caches the NLP models, classifier, and GRU model
    so training/loading occurs only once upon app startup.
    """
    # 1. Load Spacy Model
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    # 2. Train Logistic Regression Classifier
    texts = [x[0] for x in data]
    labels = [x[1] for x in data]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    clf = LogisticRegression()
    clf.fit(X, labels)

    # 3. Build Knowledge Base
    knowledge_base = build_kb(gru_data)

    # 4. Train PyTorch GRU Model
    vocab = build_vocab(gru_data)
    inv_vocab = {i: w for w, i in vocab.items()}
    vocab_size = len(vocab)
    
    model = GRUModel(vocab_size, 64, 128)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for _ in range(300):
        for text in gru_data:
            tokens = tokenize(text)
            encoded = [vocab[w] for w in tokens]
            if len(encoded) < 2:
                continue
                
            input_seq = torch.tensor(encoded[:-1]).unsqueeze(0)
            target_seq = torch.tensor(encoded[1:]).unsqueeze(0)

            output, _ = model(input_seq, None)

            loss = loss_fn(output.view(-1, vocab_size), target_seq.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return nlp, vectorizer, clf, model, vocab, inv_vocab, knowledge_base


def extract_symptoms(text: str, nlp_model) -> list[str]:
    """
    Given user text, extracts matching symptoms using simple rule-based approach
    and spaCy NLP extraction.
    """
    text = text.lower()
    
    # Clean and map common synonyms before extraction
    synonyms = {
        "diarrhoea": "diarrhea",
        "loose motion": "diarrhea",
        "stomach hurts": "abdominal pain",
        "stomach ache": "abdominal pain",
        "stomach pain": "abdominal pain",
        "belly ache": "abdominal pain",
        "tummy ache": "abdominal pain",
        "aching badly": "pain",
        "head hurts": "headache"
    }
    
    for syn, canonical in synonyms.items():
        if syn in text:
            text = text.replace(syn, canonical)

    extracted = set()

    # Rule-based matching
    for symptom in SYMPTOM_LIST:
        if symptom in text:
            extracted.add(symptom)

    # NLP-based fallback (noun chunks)
    doc = nlp_model(text)
    for chunk in doc.noun_chunks:
        for symptom in SYMPTOM_LIST:
            if symptom in chunk.text:
                extracted.add(symptom)

    return list(extracted)

def predict_disease_top3(symptoms: list[str], vectorizer, clf) -> list[str]:
    """
    Predicts the top 3 possible diseases given a list of extracted symptoms.
    """
    text = " ".join(symptoms)
    transformed_text = vectorizer.transform([text])
    probs = clf.predict_proba(transformed_text)[0]

    top3_idx = probs.argsort()[-3:][::-1]
    return [str(clf.classes_[i]) for i in top3_idx]

def rephrase_with_gru(text: str, model, vocab: dict, inv_vocab: dict, max_len=40) -> str:
    """
    Rephrases/generates output text using the trained PyTorch GRU model.
    """
    model.eval()

    # Seed the model with initial factually correct sequence
    words = tokenize(text)[:5]
    if not words:
        return "No information available."

    for _ in range(max_len):
        encoded = torch.tensor([vocab.get(w, 0) for w in words]).unsqueeze(0)
        output, _ = model(encoded, None)

        # Get probabilities of next word via softmax
        probs = torch.softmax(output[0, -1], dim=0).detach().numpy()
        next_id = np.random.choice(len(probs), p=probs)
        next_word = inv_vocab[next_id]

        if next_word == "<end>":
            break

        # Stop repetitions
        if next_word in words[-3:]:
            continue

        words.append(next_word)

    return " ".join(words).replace("<start>", "").replace("<end>", "").strip()

def get_diagnosis_response(user_text: str) -> dict:
    """
    The main pipeline function that brings it all together!
    Extracts symptoms, predicts diagnosis, and generates a rephrased response.
    """
    # Grab our pre-trained/cached models
    nlp, vectorizer, clf, model, vocab, inv_vocab, knowledge_base = load_models()
    
    # Extract symptoms from user string
    symptoms = extract_symptoms(user_text, nlp)
    
    # If no symptoms extracted, we fall back to raw input to give potential options
    used_raw_text = False
    if not symptoms:
        used_raw_text = True
        prediction_input = [user_text]
        symptoms_display = []
    else:
        prediction_input = symptoms
        symptoms_display = symptoms

    # Get diseases matching those symptoms (or raw text)
    diseases = predict_disease_top3(prediction_input, vectorizer, clf)
    disease = diseases[0]

    # Step 1: Get correct factual data
    base_text = knowledge_base.get(disease, "No data available")

    # Step 2: Generate improved response (GEN AI PART)
    generated_text = rephrase_with_gru(base_text, model, vocab, inv_vocab)

    return {
        "symptoms": symptoms_display,
        "used_raw_text": used_raw_text,
        "top3_diseases": diseases,
        "response": generated_text
    }
