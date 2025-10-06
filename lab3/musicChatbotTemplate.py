from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings


# langchain_chroma is used to create the vector store, solves the issue of importing lots of data into the model
# sentence_transformers is used to embed the text
# langchain_ollama is used to create the LLM
# langchain.chains is used to create the chain
# langchain.prompts is used to create the prompt

# Temperature is the randomness of the model, how deterministic the model is. 0 is most deterministic, 1 is most random.
llm = ChatOllama(model="phi3", temperature=0.7)
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    collection_name="musicFull",
    embedding_function=embedding_model,
    persist_directory="./chroma"
)

# k is the number of documents to retrieve
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# Prompt template is used to create the prompt for the model. We provide context, chat history and the question.
qa_prompt = PromptTemplate(
    template="""You are a knowledgeable music expert. You are given a question and a context of music reviews. You are to answer the question based on the context. You are meant to help users discover new music and learn about music.

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