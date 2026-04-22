import streamlit as st
from model import get_diagnosis_response

st.set_page_config(
    page_title="AI Medical Assistant 🩺",
    page_icon="🩺",
    layout="centered"
)

# # Sidebar with model explanation
# with st.sidebar:
#     st.title("About the Model 🧠")
#     st.info("This is an AI Medical Diagnosis Assistant designed to predict top possible diseases based on your reported symptoms.")
#     st.markdown("""
#     **How it works under the hood:**
#     1. **Symptom Extractor**: Uses rule-based matching and `spaCy` NLP (Noun Chunks) to identify symptoms from your raw text.
#     2. **Disease Classifier**: A Scikit-Learn `LogisticRegression` pipeline trained with TF-IDF vectors maps symptoms to possible diseases.
#     3. **Explanation Generator**: A custom PyTorch `GRUModel` generates a human-readable explanation based on a medical knowledge base.
#     """)
#     st.markdown("---")
#     st.caption("Built with Streamlit and PyTorch")

st.title("AI Medical Assistant 🩺")

# Strict Warning Alert
st.warning("⚠️ **Medical Disclaimer:** This application is merely an AI demonstration and a proof-of-concept. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a certified physician for medical concerns.")

# Initialize chat history into session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# React to user input in the chat
if user_input := st.chat_input("Describe your symptoms (e.g., I have a headache and chest pain)"):
    # Output the user message directly
    st.chat_message("user").markdown(user_input)
    # Persist user message in session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Process the bot response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your symptoms and predicting diseases..."):
            
            # Fetch diagnosis from backend pipeline (cached heavily to only load the model once)
            result = get_diagnosis_response(user_input)
            
            # If we couldn't extract explicit rule-based symptoms but still made a prediction from raw text
            if result.get("used_raw_text", False):
                symptoms_md = "*None explicitly mapped (using raw text for soft prediction)*"
            else:
                symptoms_md = " ".join([f"`{s}`" for s in result["symptoms"]])
                
            top_disease = result["top3_diseases"][0].title()
            other_diseases = ", ".join([d.title() for d in result["top3_diseases"][1:]])
            generated_response = result["response"]
            
            md_response = f"""
**Extracted Symptoms:** {symptoms_md}

**Primary Prediction:** 🔴 **{top_disease}**
*(Secondary predictions: {other_diseases})*

**Overview & Treatment Information:**
{generated_response.capitalize()}.

*Disclaimer: Proceed to a doctor for clinical diagnosis.*
"""
            st.markdown(md_response)
            # Persist bot response in session state
            st.session_state.messages.append({"role": "assistant", "content": md_response})
