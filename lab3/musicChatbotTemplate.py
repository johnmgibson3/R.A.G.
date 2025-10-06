from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

llm = ChatOllama(model="phi3", temperature=0.7)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="musicSmall",
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

print("🎵 Music Recommendation Chatbot")
print("Ask me about albums, artists, or get personalized recommendations!")
print("Type 'exit' or 'quit' to end the conversation.\n")

while True:
    query = input("🎤 Ask me about an album or review: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye! Keep listening to great music!")
        break
    
    result = chat_chain.invoke({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    print("\n🎧 Response:\n", answer)
 
    chat_history.append((query, answer))