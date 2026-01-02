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
GAME_SYSTEMS = {
    "coc7e": {
        "name": "Call of Cthulhu 7th Edition",
        "description": "Classic horror investigation RPG",
        "persist_directory": "./chroma_db_coc7e",
        "sources": [
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
        ],
        "example_questions": [
            "What is the sanity mechanic?",
            "How do skill checks work?",
            "What happens when an investigator goes insane?",
            "What equipment do investigators start with?",
        ]
    },
    # Add more systems here as needed
    # "dnd5e": {
    #     "name": "Dungeons & Dragons 5th Edition",
    #     "description": "Fantasy adventure RPG",
    #     "persist_directory": "./chroma_db_dnd5e",
    #     "sources": [
    #         {
    #             "path": "pdf_files/dnd5e_phb.pdf",
    #             "role": "rules",
    #             "description": "Player's Handbook"
    #         },
    #     ],
    #     "example_questions": [
    #         "How does advantage work?",
    #         "What are the character classes?",
    #     ]
    # },
}

DEFAULT_SYSTEM = "coc7e"
OLLAMA_MODEL = "fluffy/magnum-v4-9b:q5_K_M"
# OLLAMA_MODEL = "mistral:7b-instruct"

def setup_rag_system(system_config, skip_processing=False):
    """Initialize the complete RAG system for a specific game system

    Args:
        system_config: Dictionary containing system configuration (name, sources, persist_directory, etc.)
        skip_processing: If True, load existing database instead of reprocessing
    """

    system_id = system_config["id"]
    system_name = system_config["name"]
    persist_directory = system_config["persist_directory"]
    sources = system_config["sources"]

    # Create embeddings using a small, local model
    # This model runs on CPU and is free
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Check if we should load existing database or create new one
    if skip_processing and os.path.exists(persist_directory):
        print(f"📦 Loading existing vector database for {system_name}...")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print(f"   ✅ Loaded vector database from {persist_directory}\n")
    else:
        # Check for duplicate paths in sources
        seen_paths = set()
        duplicates = []
        for source in sources:
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
        if os.path.exists(persist_directory):
            print(f"🗑️  Clearing existing vector database at {persist_directory}...")
            shutil.rmtree(persist_directory)
            print("   ✅ Cleared\n")

        print(f"📄 Step 1: Loading PDFs for {system_name}...")
        # Load all PDFs and combine documents
        all_documents = []
        for source in sources:
            pdf_path = source["path"]
            role = source["role"]
            description = source["description"]

            print(f"   Loading {pdf_path} ({role})...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Add metadata to each document
            for doc in documents:
                doc.metadata["system_id"] = system_id
                doc.metadata["system_name"] = system_name
                doc.metadata["source_role"] = role
                doc.metadata["source_description"] = description
                doc.metadata["source_path"] = pdf_path

            all_documents.extend(documents)
            print(f"   Loaded {len(documents)} pages from {description}")

        print(f"   Total: {len(all_documents)} pages from {len(sources)} PDF(s)")

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
            persist_directory=persist_directory
        )
        print(f"   ✅ Vector database created at {persist_directory}\n")

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
    prompt_template = f"""You are a helpful assistant for {system_name} game masters.
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

Context: {{context}}

Question: {{question}}

Answer:"""

    prompt = ChatPromptTemplate.from_template(prompt_template)

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

def interactive_mode(rag_chain, retriever, system_name):
    """Interactive question-answering loop"""
    print("\n" + "="*70)
    print(f"🎲 {system_name} RAG Assistant")
    print("="*70)
    print(f"\nAsk questions about your {system_name} documents!")
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
    parser = argparse.ArgumentParser(description='Multi-System Tabletop RPG RAG Assistant')
    parser.add_argument('--system', type=str, default=DEFAULT_SYSTEM,
                        help=f'Game system to use (default: {DEFAULT_SYSTEM})')
    parser.add_argument('--list-systems', action='store_true',
                        help='List all available game systems and exit')
    parser.add_argument('--reprocess', action='store_true',
                        help='Reprocess PDFs and rebuild vector database (otherwise loads existing database)')
    parser.add_argument('--run-examples', action='store_true',
                        help='Run example questions before interactive mode')
    args = parser.parse_args()

    # Handle --list-systems
    if args.list_systems:
        print("\n📚 Available Game Systems:")
        print("="*70)
        for system_id, config in GAME_SYSTEMS.items():
            default_marker = " (default)" if system_id == DEFAULT_SYSTEM else ""
            print(f"\n  {system_id}{default_marker}")
            print(f"    Name: {config['name']}")
            print(f"    Description: {config['description']}")
            print(f"    Database: {config['persist_directory']}")
            print(f"    Sources: {len(config['sources'])} PDF(s)")
        print("\n" + "="*70)
        print(f"\nUsage: python simple_rag.py --system SYSTEM_ID")
        return

    # Validate selected system
    if args.system not in GAME_SYSTEMS:
        print(f"❌ Error: Unknown system '{args.system}'")
        print(f"   Available systems: {', '.join(GAME_SYSTEMS.keys())}")
        print(f"   Use --list-systems to see details")
        return

    # Get system configuration
    system_config = GAME_SYSTEMS[args.system].copy()
    system_config["id"] = args.system
    system_name = system_config["name"]
    persist_directory = system_config["persist_directory"]
    sources = system_config["sources"]

    print(f"\n🎮 Using system: {system_name}")

    # Check if we're loading existing database but it doesn't exist
    if not args.reprocess and not os.path.exists(persist_directory):
        print(f"❌ Error: Vector database not found at {persist_directory}")
        print(f"   Run with --reprocess to create the database from PDFs")
        return

    # Check if PDFs exist (only if we're reprocessing)
    if args.reprocess:
        missing_pdfs = [source for source in sources if not os.path.exists(source["path"])]
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
    rag_chain, retriever = setup_rag_system(system_config, skip_processing=not args.reprocess)

    # Check if setup was successful
    if rag_chain is None or retriever is None:
        return

    # Run some example queries if requested
    if args.run_examples:
        print("\n" + "="*70)
        print("🧪 Testing with example questions...")
        print("="*70)

        example_questions = system_config.get("example_questions", [
            "What are the core mechanics?",
            "How does combat work?",
        ])

        for q in example_questions:
            query_rag(rag_chain, retriever, q)
            input("\n⏸️  Press Enter to continue...")

    # Enter interactive mode
    interactive_mode(rag_chain, retriever, system_name)

if __name__ == "__main__":
    main()