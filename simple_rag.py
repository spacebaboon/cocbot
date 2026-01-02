"""
Simple RAG system for Call of Cthulhu PDFs - Modern LangChain
This script demonstrates the core RAG pipeline:
1. Load PDF
2. Split into chunks
3. Create embeddings and store in vector DB
4. Query with context retrieval
"""

import os
import shutil
import argparse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configuration
PDF_PATHS = [
    "pdf_files/investigator_handbook.pdf",
    "pdf_files/keepers_rulebook.pdf",
    # Add more PDFs here as needed
]
OLLAMA_MODEL = "mistral:7b-instruct"  # Change to whatever you have running
PERSIST_DIRECTORY = "./chroma_db"

def setup_rag_system(skip_processing=False):
    """Initialize the complete RAG system"""

    # Create embeddings using a small, local model
    # This model runs on CPU and is free
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Check if we should load existing database or create new one
    if skip_processing and os.path.exists(PERSIST_DIRECTORY):
        print("📦 Loading existing vector database...")
        vectorstore = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings
        )
        print(f"   ✅ Loaded vector database from {PERSIST_DIRECTORY}\n")
    else:
        # Clear existing database if it exists
        if os.path.exists(PERSIST_DIRECTORY):
            print(f"🗑️  Clearing existing vector database at {PERSIST_DIRECTORY}...")
            shutil.rmtree(PERSIST_DIRECTORY)
            print("   ✅ Cleared\n")

        print("📄 Step 1: Loading PDFs...")
        # Load all PDFs and combine documents
        all_documents = []
        for pdf_path in PDF_PATHS:
            print(f"   Loading {pdf_path}...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
            print(f"   Loaded {len(documents)} pages from {pdf_path}")

        print(f"   Total: {len(all_documents)} pages from {len(PDF_PATHS)} PDF(s)")

        print("\n✂️  Step 2: Splitting into chunks...")
        # Split documents into chunks
        # chunk_size: how many characters per chunk
        # chunk_overlap: overlap between chunks (helps maintain context)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Small chunks for testing
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_documents(all_documents)
        print(f"   Created {len(chunks)} chunks")

        # Let's see what a chunk looks like
        print(f"\n   Example chunk:")
        print(f"   {'-'*50}")
        print(f"   {chunks[0].page_content[:200]}...")
        print(f"   {'-'*50}")

        print("\n🧮 Step 3: Creating embeddings...")
        print("   Embedding model loaded")

        print("\n💾 Step 4: Storing in vector database...")
        print("   (This may take a few minutes for large PDFs...)")
        # Create the vector store
        # This will take a moment as it processes all chunks
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        print(f"   ✅ Vector database created at {PERSIST_DIRECTORY}\n")

    print("🤖 Setting up LLM...")
    # Connect to Ollama
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        base_url="http://localhost:11434",
        temperature=0.7  # Some creativity, but not too much
    )
    print(f"   Connected to {OLLAMA_MODEL}")

    print("\n🔗 Creating RAG chain...")
    # Create a custom prompt template
    # This tells the LLM how to use the retrieved context
    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant for Call of Cthulhu tabletop RPG.
Use the following context from the rulebook to answer the question.
If you don't know the answer based on the context, say so - don't make up rules.

Context: {context}

Question: {question}

Answer:""")
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
    )
    
    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the RAG chain using LCEL (LangChain Expression Language)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("   ✅ RAG system ready!\n")
    return rag_chain, retriever

def query_rag(rag_chain, retriever, question):
    """Ask a question and show the results"""
    # E5 models expect queries to be prefixed
    prefixed_question = f"query: {question}"
    
    print(f"\n❓ Question: {question}")
    print("="*70)
    
    # First, retrieve the relevant documents to show them
    print("\n📚 Retrieved Context:")
    print("-"*70)
    docs = retriever.invoke(prefixed_question)
    for i, doc in enumerate(docs, 1):
        print(f"\nChunk {i} (Page {doc.metadata.get('page', 'unknown')}):")
        content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
        print(content)
    
    # Now get the answer from the chain
    print("\n"*2 + "💡 Answer:")
    print("-"*70)
    answer = rag_chain.invoke(prefixed_question)
    print(answer)
    print("="*70)

def interactive_mode(rag_chain, retriever):
    """Interactive question-answering loop"""
    print("\n" + "="*70)
    print("🎲 Call of Cthulhu RAG Assistant")
    print("="*70)
    print("\nAsk questions about your CoC documents!")
    print("Type 'quit' or 'exit' to stop\n")
    
    while True:
        question = input("\n🎭 You: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye, Keeper!")
            break
            
        if not question:
            continue
            
        query_rag(rag_chain, retriever, question)

def main():
    """Main execution"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Call of Cthulhu RAG Assistant')
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip PDF processing and load existing vector database')
    parser.add_argument('--skip-examples', action='store_true',
                        help='Skip example questions and go straight to interactive mode')
    parser.add_argument('--interactive-only', action='store_true',
                        help='Skip both processing and examples (same as --skip-processing --skip-examples)')
    args = parser.parse_args()

    # Handle --interactive-only shortcut
    if args.interactive_only:
        args.skip_processing = True
        args.skip_examples = True

    # Check if we're skipping processing but database doesn't exist
    if args.skip_processing and not os.path.exists(PERSIST_DIRECTORY):
        print(f"❌ Error: Vector database not found at {PERSIST_DIRECTORY}")
        print(f"   You must run the script without --skip-processing first to create the database")
        return

    # Check if PDFs exist (only if we're not skipping processing)
    if not args.skip_processing:
        missing_pdfs = [pdf for pdf in PDF_PATHS if not os.path.exists(pdf)]
        if missing_pdfs:
            print(f"❌ Error: The following PDF(s) not found:")
            for pdf in missing_pdfs:
                print(f"   - {pdf}")
            print(f"   Please place your PDF files in the correct location")
            return

    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            raise Exception("Ollama not responding")
    except Exception as e:
        print(f"❌ Error: Cannot connect to Ollama!")
        print(f"   Make sure Ollama is running on Windows")
        print(f"   Test with: curl http://localhost:11434/api/tags")
        return

    # Setup the RAG system
    rag_chain, retriever = setup_rag_system(skip_processing=args.skip_processing)

    # Run some example queries to see it work (unless skipped)
    if not args.skip_examples:
        print("\n" + "="*70)
        print("🧪 Testing with example questions...")
        print("="*70)

        example_questions = [
            "What is the sanity mechanic?",
            "How do skill checks work?",
            "What happens when an investigator goes insane?",
            "Who is the main antagonist?",  # Story/narrative
            "What equipment do investigators start with?",  # Items/gear
            "What year is this set in?",  # Setting/context
            "How many players can play?"  # Meta/gameplay
        ]

        for q in example_questions:
            query_rag(rag_chain, retriever, q)
            input("\n⏸️  Press Enter to continue...")

    # Enter interactive mode
    interactive_mode(rag_chain, retriever)

if __name__ == "__main__":
    main()