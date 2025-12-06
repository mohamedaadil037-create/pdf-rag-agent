import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pypdf import PdfReader  # <-- NEW: The PDF Specialist

# 1. LOAD MODELS
print("Loading models... (This is the heavy lifting)")
embedder = SentenceTransformer('all-MiniLM-L6-v2')
bot = pipeline("text2text-generation", model="google/flan-t5-small")

# 2. READ THE PDF (The Professional Upgrade)
print("Reading the PDF...")
reader = PdfReader("data/manual.pdf")  # Make sure your file is named manual.pdf
pdf_text = ""

# Loop through every page and grab the text
for page in reader.pages:
    pdf_text += page.extract_text() + "\n"

# Split the extracted text into sentences (Simple chunking)
# Real pros split by "meaning", but splitting by "." is fine for V1
documents = [s.strip() for s in pdf_text.split('.') if len(s.strip()) > 10]

print(f"I found {len(documents)} sentences in your PDF!")
doc_vectors = embedder.encode(documents)

# 3. CHAT LOGIC (Same as before)
def chat_logic(user_message, history):
    # Search
    q_vec = embedder.encode([user_message])
    scores = cosine_similarity(q_vec, doc_vectors)
    best_index = np.argmax(scores)
    best_info = documents[best_index]
    
    # Generate
    full_prompt = f"Context: {best_info} Question: {user_message}"
    reply = bot(full_prompt, max_length=100)[0]['generated_text']
    return reply

# 4. LAUNCH UI
demo = gr.ChatInterface(
    fn=chat_logic, 
    title="PDF Chatbot Professional 🤖", 
    description="I read the PDF in the data folder. Ask me anything about it!"
)

if __name__ == "__main__":
    demo.launch()