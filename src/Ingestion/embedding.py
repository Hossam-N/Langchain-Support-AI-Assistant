import json
from pathlib import Path
from langchain_community.vectorstores import Chroma 
from langchain_huggingface import HuggingFaceEmbeddings


class Embedder:
    def __init__(self, persist_directory = "../data/vectorstore", model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory 
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def load_chunks(self, input_path):
        with open(input_path, "r", encoding="utf-8") as f: 
            return json.load(f)

    def build_vectorstore(self, chunks):
        texts = [c["content"] for c in chunks]
        metadatas = [{"url" : c["url"] , "chunk_id" : c["chunk_id"]} for c in chunks]
        vectorstore = Chroma.from_texts( 
          texts=texts, 
          embedding=self.embedding_model, 
          metadatas=metadatas, 
          persist_directory=self.persist_directory) 
        vectorstore.persist() 
        print(f"[INFO] Vectorstore built & persisted at {self.persist_directory}") 
        return vectorstore


if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(description="Build embeddings for preprocessed chunks.") 
    parser.add_argument("--input", required=True, help="Path to preprocessed chunks JSON") 
    parser.add_argument("--output", default="../data/vectorstore", help="Directory to persist vectorstore")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="HF embedding model name")
    
    args = parser.parse_args()
    embedder = Embedder(persist_directory=args.output, model_name=args.model)
    chunks = embedder.load_chunks(args.input)
    embedder.build_vectorstore(chunks)




