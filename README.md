# RAG - Retrieval Augmented Generation

- THIS PROJECT is a music recomendation system. ex. you could ask the system to play you sad songs because your GF broke up with you
- User prompts are submitted to a retrieval system. Prompts are converted to vector embeddings which run through a pre-processed knowledge repo, which then returns the most relavent results.
- AKA this process hydrates your LLM with relavent info

**LAB STEPS:**

Step 1: Embedding music documents in ChromaDB via chunking

Step 2: Building a CLI Music Recommender Chatbot

Step 3: Rapid prototyping with Gradio

Step 4: Initial evaluation using different models, prompts, temperatures, top-k in your retriever, and size of ChromaDB



**CLASS NOTES**

**Retrievers** - 

* Sparse (i.e., keyword) vs Den
  * BM25 (Best Matching 25)
    * Uses TF-IDF (occurrences of terms in documents and acorss the document corpora)
    * Does not capture semantic meaning but easily interpretable
    * Typicaly considered baseline retrieval (i.e., OG)
* Dense - more detailed
  * Encoders

EX. How they search

* Command: "Find all documents that relate to dogs and rivers"
  * Sparse:
    * Dog, dogs
    * River, river
  * Dense:
    * Canines, Duchshaund, puppies, Old Yeller
    * Seine, Mississippi, irrigation, pond
  * Hybrid: Dense + Sparse!
    * weighted sum or rank fusion (You choose weights)
    * *Ensemble retrievers* in LangChain

**Chunking Strategies -** how we split of documents and text

* Recursive - Natural boundries until size constraint is met. Focuses on paragraphs then sentence, letters top down
* Fixed-size
* Semantic
* Overlapping (sliding) -  Chunks overlap to keep
* Fixed-size
* Embedding-based
* Metadata-aware

**Multi-modal Documents**

* Determine if a user prompt is asking for text, image, or audio output?
* Each type of document has their own encoders
  * text, image, audio, video, etc.
  * Joint text.images embeddings models
    * CLIP by OpenAI
    * ALIGN by Google
*
