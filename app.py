# Importing required libraries
from datasets import load_dataset
import requests
import json

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
                "num_predict": 50,  # Reduced for faster response
                "temperature": 0.7,
            }
        }
        
        print(f"Calling Ollama with model: {model}")
        print("This may take a moment...")
        
        try:
            # Increased timeout for first run (model loading)
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "No response generated")
        except requests.exceptions.Timeout:
            return "Ollama request timed out. The model might be loading for the first time. Please try again."
        except requests.exceptions.ConnectionError:
            return "Cannot connect to Ollama. Make sure 'ollama serve' is running."
        except requests.exceptions.RequestException as e:
            return f"Error calling Ollama: {str(e)}"
    
    def generate_answer(self, question):
        """Generate answer using RAG approach"""
        print(f"Retrieving documents for: {question}")
        
        # Retrieve relevant documents
        relevant_docs = self.retrieve_documents(question)
        
        if not relevant_docs:
            return "No relevant documents found."
        
        # Build context from retrieved documents
        context = "\n\n".join(relevant_docs)
        
        # Create prompt
        prompt = f"""Given the following information, answer the question.

Context:
{context}

Question: {question}
Answer:"""
        
        print("Generating answer with Ollama...")
        # Generate answer using Ollama
        answer = self.call_ollama(prompt)
        return answer

# Main execution
if __name__ == "__main__":
    print("Initializing Simple RAG system...")
    
    # First check if Ollama is accessible
    try:
        test_response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if test_response.status_code == 200:
            models = test_response.json()
            print("Available Ollama models:")
            for model in models.get('models', []):
                print(f"  - {model.get('name', 'Unknown')}")
        else:
            print("Ollama server responded but with error. Please check 'ollama serve'")
    except requests.exceptions.RequestException:
        print("⚠️  Ollama server not accessible!")
        print("Please run these commands in separate terminals:")
        print("1. ollama serve")
        print("2. ollama pull mistral")
        print("Then run this script again.")
        exit(1)
    
    try:
        rag = SimpleRAG()
        
        # Test question
        question = "What does Rhodes Statue look like?"
        print(f"\nQuestion: {question}")
        print("=" * 50)
        
        answer = rag.generate_answer(question)
        print(f"Answer: {answer}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Make sure Mistral model is installed: ollama pull mistral")
        print("3. Check your internet connection for dataset download")
        print("4. Try installing: pip install haystack-ai datasets requests")