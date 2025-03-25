from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from dev_tools.enums.embeddings import LocalMPNetBaseV2

# Initialize the embedding function
embedding_function = LocalMPNetBaseV2().ef

# Create a new Chroma instance using Langchain
db = Chroma(
    persist_directory="/Users/dabbos/mlv0rr/MoEtonomous/.dev.sess/.chroma_db",
    embedding_function=embedding_function,
    collection_name="arxiv_daily"
)

# Example of adding documents
documents = [
    Document(
        page_content="Example document content",
        metadata={"source": "example.txt"}
    )
]

# Add documents to the vector store
db.add_documents(documents)

# Example of similarity search
query = "Your search query here"
docs = db.similarity_search(query)

# Example of using as a retriever
retriever = db.as_retriever(search_kwargs={"k": 5})
retrieved_docs = retriever.get_relevant_documents(query)

# Example of similarity search with score
docs_and_scores = db.similarity_search_with_score(query)

# Print results
for doc, score in docs_and_scores:
    print(f"Content: {doc.page_content}")
    print(f"Score: {score}")
    print(f"Metadata: {doc.metadata}\n") 