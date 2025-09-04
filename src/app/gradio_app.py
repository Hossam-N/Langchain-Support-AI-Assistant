import gradio as gr
from src.rag.pipeline import RAGPipeline


rag = RAGPipeline(
    persist_directory="data/vectorstore",
    embed_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="google/flan-t5-base"
)


def qa_fn(query):
    try:
        answer, sources = rag.ask(query)
        sources_text = "\n".join([f"- {s['url']}" for s in sources])
        return answer, sources_text
    except Exception as e:
        return f"Error: {e}", ""


with gr.Blocks() as demo:
    with gr.Row():
        query = gr.Textbox(label="Your Question", placeholder="Type your question here...")
    
    with gr.Row():
        answer = gr.Textbox(label="Answer", interactive=False)
        sources = gr.Textbox(label="Sources", interactive=False)

    submit = gr.Button("Ask")

    submit.click(fn=qa_fn, inputs=query, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=True)






