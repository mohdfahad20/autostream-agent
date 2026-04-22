"""
rag_pipeline.py
---------------
Loads the AutoStream knowledge base (Markdown ONLY) and builds a FAISS
vector store for retrieval using HuggingFace embeddings.

OPTIMISATION:
- First run  → builds + saves FAISS index
- Next runs  → loads instantly from disk (⚡ fast startup)
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ fixed import
from langchain.schema import Document


# ── Paths ─────────────────────────────────────────────────────────────────────
KB_PATH         = Path(__file__).parent / "knowledge_base" / "autostream_kb.md"
FAISS_INDEX_DIR = Path(__file__).parent / "faiss_index"


# ── Shared embedding model (loaded once) ──────────────────────────────────────
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def build_retriever(k: int = 3):
    """
    Return a FAISS retriever.

    - First run  → builds index + saves to disk
    - Later runs → loads from disk (fast, no re-embedding)
    """

    # 🔥 FAST PATH → load cached index
    if FAISS_INDEX_DIR.exists():
        print("⚡ Loading FAISS index from disk (cached)...")

        vectorstore = FAISS.load_local(
            str(FAISS_INDEX_DIR),
            EMBEDDINGS,
            allow_dangerous_deserialization=True,  # required for local load
        )

        print("✅ FAISS index loaded from cache\n")
        return vectorstore.as_retriever(search_kwargs={"k": k})

    # 🧱 FIRST RUN → build index
    print(f"🚀 Building FAISS index from: {KB_PATH.name}")
    assert KB_PATH.exists(), f"❌ Knowledge base not found at {KB_PATH}"

    # 1. Load Markdown
    loader = TextLoader(str(KB_PATH), encoding="utf-8")
    raw_docs = loader.load()

    source = raw_docs[0].metadata.get("source", KB_PATH)
    print(f"📄 Source confirmed: {source}")

    # 🛑 Safety check (avoid JSON mistake)
    first_line = raw_docs[0].page_content.strip().splitlines()[0]
    assert first_line.startswith("#"), (
        f"❌ KB file does not look like Markdown! First line: '{first_line}'"
    )

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=60,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"✅ Split into {len(chunks)} chunks")

    # 3. Embed
    print("🔨 Embedding chunks (only happens once)...")
    vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)

    # 4. Save index
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_INDEX_DIR))

    print(f"💾 FAISS index saved to ./{FAISS_INDEX_DIR.name}/")
    print("✅ FAISS vector store ready\n")

    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve_context(query: str, retriever) -> str:
    """
    Retrieve relevant KB chunks for a query.
    """
    docs: list[Document] = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the knowledge base."

    return "\n\n---\n\n".join(doc.page_content.strip() for doc in docs)


# ── Smoke-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = build_retriever()

    test_queries = [
        "What is the price of the Pro plan?",
        "Do you offer refunds?",
        "What is included in the Basic plan?",
    ]

    for q in test_queries:
        print(f"🔎 Query: {q}")
        print(retrieve_context(q, retriever))
        print("=" * 60 + "\n")