import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient


# This is building the knowledge base to use for the RAG model
# Run this file to build the chroma database
df = pd.read_excel("Music Reviews Full.xlsx")

grouped_reviews = df.groupby(['Artist', 'Album', 'Release Year', 'RYM Rating', 'RYM Ratings', 'Overall ranking', 'Genres', 'Descriptors'])['Review'].apply(list).reset_index()

#split the text into chunks using langchain recursive character text splitter, can play with limit. Chunk_overlap is sliding window
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

texts = []
metadatas = []

for _, row in df.iterrows():
    chunks = splitter.split_text(row["Review"])
    metadata = {
        "artist": row["Artist"],
        "album": row["Album"],
        "year": row['Release Year'],
        "rating": row['RYM Rating'],
        "numRatings": row['RYM Ratings'],
        "rank": row['Overall ranking'],
        "genres": row['Genres'],
        "descriptors": row['Descriptors']
    }
    for chunk in chunks:
        texts.append(chunk)
        metadatas.append(metadata)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)

#Persistent client is used to store the data in a local database which can be accessed later
client = chromadb.PersistentClient(
    path="./chroma",
    settings=Settings(anonymized_telemetry=False)
)

collection = client.create_collection(name="musicFull")

BATCH_SIZE = 1000

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_metadatas = metadatas[i:i + BATCH_SIZE]
    
    # Generate embeddings for this batch
    batch_embeddings = model.encode(batch_texts)
    
    # Need to batch documents so you can feed them to the model in batches
    # Add batch to collection
    collection.add(
        documents=batch_texts,
        embeddings=batch_embeddings.tolist(),  # Convert numpy array to list
        metadatas=batch_metadatas,
        ids=[f"chunk_{j}" for j in range(i, i + len(batch_texts))]
    )
    
    print(f"✓ Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")

print("✅ All documents added successfully!")
