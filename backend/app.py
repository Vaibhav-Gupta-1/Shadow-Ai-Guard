import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="ShadowAI Guard Security Backend")

# Enable Cross-Origin Resource Sharing (CORS) so the extension can talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Cybersecurity Rule Engine (Regex Patterns)
PHONE_REGEX = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
API_KEY_REGEX = r'(?:key|secret|token|passwd|password)(?:[\s|\'|\"]*[:|=][\s|\'|\"]*)([a-zA-Z0-9_\-]{16,50})'
EMAIL_REGEX = r'[\w\.-]+@[\w\.-]+\.\w+'

# Common greetings and short conversational words to bypass NER false positives
SAFE_WHITELIST = {"hi", "hii", "hiii", "hello", "hey", "heyy", "test", "clear", "ok", "okay", "yes", "no"}

# 2. Machine Learning Named Entity Recognition (NER) Pipeline
print("🤖 Initializing Machine Learning NER Engine...")
# This will download a lightweight BERT model on its first run
ner_engine = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print("✅ ML Engine Loaded Successfully!")

class PromptModel(BaseModel):
    text: str

@app.post("/analyze")
async def analyze_prompt(payload: PromptModel):
    prompt = payload.text
    
    # Pre-processing: Clean text to evaluate against our whitelist
    cleaned_prompt = prompt.strip().lower()
    
    # Early Exit: Skip heavy scanning if it's just a common greeting or under 4 characters
    if len(cleaned_prompt) <= 3 or cleaned_prompt in SAFE_WHITELIST:
        return {"is_sensitive": False, "reason": "Safe conversation opener"}

    # Tier 1: Instant Regex Signature Checks
    if re.search(API_KEY_REGEX, prompt, re.IGNORECASE):
        return {"is_sensitive": True, "reason": "Potential API Key / Authentication Credentials detected."}
        
    if re.search(PHONE_REGEX, prompt):
        return {"is_sensitive": True, "reason": "Protected PII Detected (Phone Number match)."}
        
    if re.search(EMAIL_REGEX, prompt):
        return {"is_sensitive": True, "reason": "Protected PII Detected (Email Address match)."}

    # Tier 2: Deep Contextual Analysis via NLP Transformer
    try:
        entities = ner_engine(prompt)
        for entity in entities:
            # If the AI detects an Organization (ORG) or Person name (PER) with high confidence
            if entity['entity_group'] in ['ORG', 'PER'] and entity['score'] > 0.88:
                return {
                    "is_sensitive": True, 
                    "reason": f"Contextual corporate data leak risk ({entity['entity_group']}: '{entity['word']}')"
                }
    except Exception as e:
        print(f"ML Processing Error: {e}")

    # Clear to pass if no security rules trip
    return {"is_sensitive": False, "reason": "Safe"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)