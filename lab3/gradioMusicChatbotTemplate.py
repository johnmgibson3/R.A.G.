from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import gradio as gr

llm = ChatOllama(model="phi3", temperature=0.4)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="musicFull",
    embedding_function=embedding_model,
    persist_directory="./chroma"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

qa_prompt = PromptTemplate(
    template="""

Context from reviews:
{context}

Chat History:
{chat_history}

User question: {question}

Please provide a helpful response based on the music reviews and context available:""",
    input_variables=["context", "chat_history", "question"]
)

chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True  # optional, for debugging
)

chat_history = []

def get_response(message, history):

    # Run the chain
    result = chat_chain.invoke({
        "question": message, 
        "chat_history": chat_history
    })
    
    answer = result["answer"]
      
    # Update chat history
    chat_history.append((message, answer))

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history

def clear_history():
    """Clear chat history."""
    global chat_history
    chat_history = []
    return None

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Music Recommendation Chatbot") as demo:
    
    gr.Markdown("""
    # 🎵 Music Recommendation Chatbot
    Ask me about albums, artists, genres, or get personalized music recommendations!
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                type="messages",
                avatar_images=(None, "🎧"),
                label="Chat"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about an album, artist, or request recommendations...",
                    show_label=False,
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button("Send 🎤", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat 🗑️")
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 💡 Try asking:
            - "Recommend albums similar to Radiohead"
            - "What are some good jazz albums?"
            - "Tell me about [specific album]"
            - "I want upbeat indie rock recommendations"
            - "What's a good album for studying?"
            """)
            
            gr.Markdown("""
            ### ℹ️ About
            This chatbot uses a RAG system with:
            - Local ChromaDB vector storage
            - Music review embeddings
            - Mistral LLM via Ollama
            """)
    
    # Event handlers
    msg.submit(
        get_response, 
        inputs=[msg, chatbot], 
        outputs=chatbot
    ).then(
        lambda: "", 
        outputs=msg
    )
    
    submit_btn.click(
        get_response, 
        inputs=[msg, chatbot], 
        outputs=chatbot
    ).then(
        lambda: "", 
        outputs=msg
    )
    
    clear_btn.click(
        clear_history, 
        outputs=chatbot
    )
    
    # Example queries
    gr.Examples(
        examples=[
            "What are some critically acclaimed albums from 2023?",
            "Recommend me some ambient electronic music",
            "Tell me about albums with great guitar work",
            "I'm in the mood for melancholic indie folk",
            "What are the best albums for a road trip?"
        ],
        inputs=msg,
        label="Example Questions"
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",  # Makes it accessible on your network
        server_port=7860
    )