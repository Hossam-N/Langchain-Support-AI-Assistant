import re 
import json
from pathlib import Path
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter


class Preprocessor:
    def __init__(self, chunk_size = 1000, chunk_overlap = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators = ["\n\n", "\n", ".", "!", "?", " ", ""]
        )

    def clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def process_docs(self, docs):
        all_chunks = []
        for doc in docs:
            cleaned = self.clean_text(doc[1])
            chunks = self.splitter.split_text(cleaned)
            for i, chunk in enumerate(chunks):
                all_chunks.append(
                    {
                        "url": doc[0],
                        "chunk_id": i,
                        "content": chunk
                    }
                )
        return all_chunks

    @staticmethod
    def save(chunks, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess scraped docs into chunks.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/langchain_docs.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/langchain_chunks.json"
    )
    args = parser.parse_args()
    
    with open(args.input,"r", encoding="utf-8") as f:
        docs = json.load(f)

    preprocessor = Preprocessor(chunk_size=800, chunk_overlap=100)
    chunks = preprocessor.process_docs(docs)
    Preprocessor.save(chunks, args.output)    
    




