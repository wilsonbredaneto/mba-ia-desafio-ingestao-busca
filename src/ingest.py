import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector

for k in ("PG_VECTOR_COLLECTION_NAME","DATABASE_URL", "PDF_PATH"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")

PDF_PATH = os.getenv("PDF_PATH")
AIKEY = os.getenv("OPENAI_API_KEY")
AIMODEL = os.getenv("OPENAI_EMBEDDING_MODEL")

def ingest_pdf():

    file_dir = Path(PDF_PATH).parent / "document.pdf"
    docs = PyPDFLoader(str(file_dir)).load()
    splits = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150, add_start_index=False).split_documents(docs)
    if not splits: raise SystemExit(1)

    enriched = [
        Document(
                    page_content=d.page_content,
                    metadata={k: v for k, v in d.metadata.items() if v not in ("", None)}
                )
    for d in splits]    
    ids = [f"doc-{i}" for i in range(len(enriched))]

    embeddings = OpenAIEmbeddings(model=AIMODEL)
    store = PGVector(
        embeddings=embeddings,
        collection_name=os.getenv("PG_VECTOR_COLLECTION_NAME"),
        connection=os.getenv("DATABASE_URL"),
        use_jsonb=True,
        )
    store.add_documents(documents=enriched, ids=ids)
    pass


if __name__ == "__main__":
    ingest_pdf()