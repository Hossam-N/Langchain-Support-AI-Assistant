from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


class RAGPipeline:
    def __init__(self, persist_directory="../data/vectorstore", 
                 embed_model="sentence-transformers/all-MiniLM-L6-v2",
                 llm_model="google/flan-t5-base"):
        """
        Initialize retriever + free HuggingFace LLM pipeline.
        """
        # Load embeddings
        embedding_model = HuggingFaceEmbeddings(model_name=embed_model)

        # Load ChromaDB
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Load free HuggingFace LLM (flan-t5-base)
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
        model = AutoModelForSeq2SeqLM.from_pretrained(llm_model)

        pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
        self.llm = HuggingFacePipeline(pipeline=pipe)

        # Retrieval + QA chain
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=True
        )

    def ask(self, query: str):
        result = self.qa.invoke({"query": query})
        return result["result"], [doc.metadata for doc in result["source_documents"]]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("--db", default="../data/vectorstore", help="Vectorstore directory")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model")
    parser.add_argument("--llm_model", default="google/flan-t5-base", help="HuggingFace LLM model")

    args = parser.parse_args()

    rag = RAGPipeline(
        persist_directory=args.db,
        embed_model=args.embed_model,
        llm_model=args.llm_model
    )

    answer, sources = rag.ask(args.query)

    print("\n[ANSWER]")
    print(answer)

    print("\n[SOURCES]")
    for s in sources:
        print(s)
