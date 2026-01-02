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
PDF_SOURCES = [
    {
        "path": "pdf_files/investigator_handbook.pdf",
        "role": "rules",
        "description": "Call of Cthulhu Investigator Handbook - Core rules for character creation and gameplay"
    },
    {
        "path": "pdf_files/keepers_rulebook.pdf",
        "role": "rules",
        "description": "Call of Cthulhu Keeper's Rulebook - Game master rules and guidance"
    },
    {
        "path": "pdf_files/HOTOO_1.pdf",
        "role": "adventure",
        "description": "Horror on the Orient Express - Book 1: Campaign Book"
    },
    {
        "path": "pdf_files/HOTOO_2.pdf",
        "role": "adventure",
        "description": "Horror on the Orient Express - Book 2: Through the Alps"
    },
    {
        "path": "pdf_files/HOTOO_3.pdf",
        "role": "adventure",
        "description": "Horror on the Orient Express - Book 3: Italy and Beyond"
    },
    {
        "path": "pdf_files/HOTOO_4.pdf",
        "role": "adventure",
        "description": "Horror on the Orient Express - Book 4: Constantinople and Consequences"
    },
    {
        "path": "pdf_files/HOTOO_5.pdf",
        "role": "adventure",
        "description": "Horror on the Orient Express - Book 5: Strangers on the Train"
    },
]
OLLAMA_MODEL = "fluffy/magnum-v4-9b:q5_K_M"  
# OLLAMA_MODEL = "mistral:7b-instruct"  
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
        # Check for duplicate paths in PDF_SOURCES
        seen_paths = set()
        duplicates = []
        for source in PDF_SOURCES:
            path = source["path"]
            if path in seen_paths:
                duplicates.append(path)
            seen_paths.add(path)

        if duplicates:
            print(f"❌ Error: Duplicate PDF paths detected in PDF_SOURCES:")
            for dup in duplicates:
                print(f"   - {dup}")
            print(f"   Please remove duplicates from the configuration")
            return None, None

        # Clear existing database if it exists
        if os.path.exists(PERSIST_DIRECTORY):
            print(f"🗑️  Clearing existing vector database at {PERSIST_DIRECTORY}...")
            shutil.rmtree(PERSIST_DIRECTORY)
            print("   ✅ Cleared\n")

        print("📄 Step 1: Loading PDFs...")
        # Load all PDFs and combine documents
        all_documents = []
        for source in PDF_SOURCES:
            pdf_path = source["path"]
            role = source["role"]
            description = source["description"]

            print(f"   Loading {pdf_path} ({role})...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Add metadata to each document
            for doc in documents:
                doc.metadata["source_role"] = role
                doc.metadata["source_description"] = description
                doc.metadata["source_path"] = pdf_path

            all_documents.extend(documents)
            print(f"   Loaded {len(documents)} pages from {description}")

        print(f"   Total: {len(all_documents)} pages from {len(PDF_SOURCES)} PDF(s)")

        print("\n✂️  Step 2: Splitting into chunks...")
        # Split documents into chunks
        # chunk_size: how many characters per chunk
        # chunk_overlap: overlap between chunks (helps maintain context)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Larger chunks for more context
            chunk_overlap=100,
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
        temperature=1.0
    )
    print(f"   Connected to {OLLAMA_MODEL}")

    print("\n🔗 Creating RAG chain...")
    # Create a custom prompt template
    # This tells the LLM how to use the retrieved context
    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant for Call of Cthulhu tabletop RPG game masters.
Use the following context to answer the question. Each piece of context includes metadata about its source.

Context may come from different types of sources:
- RULES sources contain game mechanics, character creation, skill checks, and system rules
- ADVENTURE sources contain scenarios, plots, NPCs, and story content

When answering:
- Provide detailed, comprehensive answers using all relevant information from the context
- For rules questions, prioritize information from RULES sources and explain mechanics thoroughly with examples when available
- For story/scenario questions, use ADVENTURE sources and provide complete details about NPCs, locations, plot points, and encounters
- Include specific page references, stat blocks, or other concrete details from the context
- If multiple sources provide relevant information, synthesize them into a complete answer
- If you don't know the answer based on the context, say so - don't make up information

Context: {context}

Question: {question}

Answer:""")

    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10}  # Retrieve top 5 most relevant chunks
    )

    # Helper function to format documents with metadata
    def format_docs(docs):
        formatted = []
        for doc in docs:
            role = doc.metadata.get('source_role', 'unknown').upper()
            description = doc.metadata.get('source_description', 'Unknown source')
            page = doc.metadata.get('page', 'unknown')
            formatted.append(f"[{role} - {description} (Page {page})]\n{doc.page_content}")
        return "\n\n".join(formatted)
    
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
        role = doc.metadata.get('source_role', 'unknown').upper()
        description = doc.metadata.get('source_description', 'Unknown source')
        page = doc.metadata.get('page', 'unknown')
        print(f"\nChunk {i} [{role}] - {description} (Page {page}):")
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
    parser.add_argument('--reprocess', action='store_true',
                        help='Reprocess PDFs and rebuild vector database (otherwise loads existing database)')
    parser.add_argument('--run-examples', action='store_true',
                        help='Run example questions before interactive mode')
    args = parser.parse_args()

    # Check if we're loading existing database but it doesn't exist
    if not args.reprocess and not os.path.exists(PERSIST_DIRECTORY):
        print(f"❌ Error: Vector database not found at {PERSIST_DIRECTORY}")
        print(f"   Run with --reprocess to create the database from PDFs")
        return

    # Check if PDFs exist (only if we're reprocessing)
    if args.reprocess:
        missing_pdfs = [source for source in PDF_SOURCES if not os.path.exists(source["path"])]
        if missing_pdfs:
            print(f"❌ Error: The following PDF(s) not found:")
            for source in missing_pdfs:
                print(f"   - {source['path']} ({source['description']})")
            print(f"   Please place your PDF files in the correct location")
            return

    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            raise Exception("Ollama not responding")
    except Exception:
        print(f"❌ Error: Cannot connect to Ollama!")
        print(f"   Make sure Ollama is running on Windows")
        print(f"   Test with: curl http://localhost:11434/api/tags")
        return

    # Setup the RAG system
    rag_chain, retriever = setup_rag_system(skip_processing=not args.reprocess)

    # Check if setup was successful
    if rag_chain is None or retriever is None:
        return

    # Run some example queries if requested
    if args.run_examples:
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