# Importing required libraries
from datasets import load_dataset
import requests
import json
import gradio as gr

# Check your Haystack version first
try:
    import haystack
    print(f"Haystack version: {haystack.__version__}")
except:
    print("Could not determine Haystack version")

# Try different import patterns based on version
try:
    # For newer versions
    from haystack import Document
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.retrievers import InMemoryBM25Retriever
    print("Using newer Haystack API")
except ImportError:
    try:
        # For older versions
        from haystack import Document
        from haystack.document_stores import InMemoryDocumentStore
        from haystack.nodes import BM25Retriever
        print("Using older Haystack API")
    except ImportError:
        print("Could not import Haystack components")
        exit(1)

# Simple RAG implementation without Pipeline class
class SimpleRAG:
    def __init__(self):
        # Load dataset and create documents
        print("Loading dataset...")
        dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
        self.docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
        
        # Initialize document store
        print("Setting up document store...")
        try:
            # Try newer API first
            self.document_store = InMemoryDocumentStore()
            self.document_store.write_documents(self.docs)
            
            # Try newer retriever
            try:
                from haystack.components.retrievers import InMemoryBM25Retriever
                self.retriever = InMemoryBM25Retriever(self.document_store)
                self.retriever_type = "new"
            except ImportError:
                from haystack.nodes import BM25Retriever
                self.retriever = BM25Retriever(document_store=self.document_store)
                self.retriever_type = "old"
                
        except Exception as e:
            print(f"Error setting up document store: {e}")
            # Fallback to simple list storage
            self.docs_simple = [doc.content for doc in self.docs]
            self.retriever = None
            self.retriever_type = "simple"
    
    def simple_search(self, query, top_k=3):
        """Simple keyword-based search fallback"""
        query_words = query.lower().split()
        scored_docs = []
        
        for i, doc_content in enumerate(self.docs_simple):
            score = sum(1 for word in query_words if word in doc_content.lower())
            if score > 0:
                scored_docs.append((score, doc_content))
        
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:top_k]]
    
    def retrieve_documents(self, query, top_k=3):
        """Retrieve relevant documents"""
        if self.retriever_type == "new":
            try:
                result = self.retriever.run(query=query, top_k=top_k)
                return [doc.content for doc in result["documents"]]
            except:
                return self.simple_search(query, top_k)
        elif self.retriever_type == "old":
            try:
                result = self.retriever.retrieve(query=query, top_k=top_k)
                return [doc.content for doc in result]
            except:
                return self.simple_search(query, top_k)
        else:
            return self.simple_search(query, top_k)
    
    def call_ollama(self, prompt, model="mistral"):
        """Call Ollama API with better error handling"""
        # First check if Ollama is running
        try:
            health_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if health_response.status_code != 200:
                return "Ollama server is not responding. Please run 'ollama serve' in a terminal."
        except requests.exceptions.RequestException:
            return "Ollama server is not running. Please run 'ollama serve' in a terminal."
        
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": 100,  # Increased for better answers in UI
                "temperature": 0.7,
            }
        }
        
        try:
            # Timeout for UI (2 minutes)
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response generated")
        except requests.exceptions.Timeout:
            return "⏱️ Request timed out. The model might be loading. Please try again."
        except requests.exceptions.ConnectionError:
            return "❌ Cannot connect to Ollama. Make sure 'ollama serve' is running."
        except requests.exceptions.RequestException as e:
            return f"❌ Error: {str(e)}"
    
    def generate_answer(self, question, model="mistral", num_docs=3):
        """Generate answer using RAG approach"""
        if not question.strip():
            return "Please enter a question."
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_documents(question, top_k=num_docs)
        
        if not relevant_docs:
            return "No relevant documents found for your question."
        
        # Build context from retrieved documents
        context = "\n\n".join(relevant_docs)
        
        # Create prompt
        prompt = f"""Given the following information about the Seven Wonders of the Ancient World, answer the question accurately and concisely.

Context:
{context}

Question: {question}
Answer:"""
        
        # Generate answer using Ollama
        answer = self.call_ollama(prompt, model=model)
        return answer

# Initialize RAG system
print("Initializing RAG system...")
try:
    rag_system = SimpleRAG()
    print("✅ RAG system initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing RAG system: {e}")
    rag_system = None

def check_ollama_status():
    """Check if Ollama is running and return available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json()
            model_names = [model.get('name', 'Unknown') for model in models.get('models', [])]
            return True, model_names
        else:
            return False, []
    except:
        return False, []

def answer_question(question, model_name, num_documents):
    """Main function for Gradio interface"""
    if not rag_system:
        return "❌ RAG system not initialized. Please check the console for errors."
    
    if not question.strip():
        return "Please enter a question about the Seven Wonders of the Ancient World."
    
    # Check Ollama status
    is_running, available_models = check_ollama_status()
    if not is_running:
        return "❌ Ollama server is not running. Please start it with 'ollama serve' command."
    
    if model_name not in [model.split(':')[0] for model in available_models]:
        return f"❌ Model '{model_name}' not found. Available models: {', '.join(available_models)}"
    
    return rag_system.generate_answer(question, model=model_name, num_docs=num_documents)

# Sample questions for easy testing
sample_questions = [
    "What does the Rhodes Statue look like?",
    "How tall is the Great Pyramid of Giza?",
    "Where was the Lighthouse of Alexandria built?",
    "What happened to the Colossus of Rhodes?",
    "Who built the Hanging Gardens of Babylon?",
    "What was the Temple of Artemis used for?",
    "How was the Mausoleum at Halicarnassus destroyed?"
]

# Create Gradio interface
with gr.Blocks(title="Seven Wonders RAG System", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🏛️ Seven Wonders RAG System
        
        Ask questions about the **Seven Wonders of the Ancient World** and get AI-powered answers!
        
        This system uses Retrieval-Augmented Generation (RAG) to find relevant information and generate accurate responses.
        
        **Make sure Ollama is running:** `ollama serve`
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask anything about the Seven Wonders...",
                lines=2
            )
            
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=["mistral", "llama3.2:1b", "llama3.2", "llama3.1"],
                    value="mistral",
                    label="AI Model",
                    info="Choose the AI model to use"
                )
                
                num_docs_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Number of Documents",
                    info="How many relevant documents to use"
                )
            
            submit_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("### 💡 Sample Questions:")
            for i, sample in enumerate(sample_questions):
                gr.Button(
                    sample,
                    size="sm"
                ).click(
                    fn=lambda x=sample: x,
                    outputs=question_input
                )
    
    answer_output = gr.Textbox(
        label="Answer",
        lines=8,
        max_lines=15,
        interactive=False,
        show_copy_button=True
    )
    
    # Status indicator
    with gr.Row():
        status_text = gr.Markdown("**Status:** Ready to answer questions!")
    
    def update_status():
        is_running, models = check_ollama_status()
        if is_running:
            return f"**Status:** ✅ Ollama running | Available models: {', '.join(models)}"
        else:
            return "**Status:** ❌ Ollama not running - Please start with 'ollama serve'"
    
    # Update status on load
    demo.load(fn=update_status, outputs=status_text)
    
    # Submit button action
    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, model_dropdown, num_docs_slider],
        outputs=answer_output
    )
    
    # Enter key submission
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, model_dropdown, num_docs_slider],
        outputs=answer_output
    )

if __name__ == "__main__":
    # Check Ollama status before launching
    is_running, models = check_ollama_status()
    if not is_running:
        print("⚠️  Warning: Ollama server not detected!")
        print("Please run 'ollama serve' in another terminal before using the interface.")
    else:
        print(f"✅ Ollama detected with models: {', '.join(models)}")
    
    print("🚀 Launching Gradio interface...")
    demo.launch(
        server_name="127.0.0.1",  # Local access only
        server_port=7860,
        share=False,  # Set to True if you want a public link
        show_error=True,
        inbrowser=True  # Automatically opens browser
    )