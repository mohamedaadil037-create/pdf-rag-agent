# # 🤖 PDF RAG Chatbot

### How it works:
1. **Ingests** a PDF file.
2. **Embeds** the text using `all-MiniLM-L6-v2`.
3. **Retrieves** relevant context using Cosine Similarity.
4. **Generates** answers using `google/flan-t5-small`.

*Built with Gradio, Transformers, and PyTorch.*
