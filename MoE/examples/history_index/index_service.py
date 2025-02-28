from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class IndexService:

    def __init__(self, persist_directory, embedding_fn):
        """Initializes the Chroma vector store."""
        self.persist_directory = persist_directory
        self.embedding_fn = embedding_fn
        self.vectorstores = {}

    def _get_vectorstore(self, index_key: str):
        """Retrieves or initializes a vector store for a specific index key."""
        if index_key not in self.vectorstores:
            self.vectorstores[index_key] = Chroma(persist_directory=f"{self.persist_directory}/{index_key}", embedding_function=self.embedding_fn)
        return self.vectorstores[index_key]

    def load_document(self, document: Document, index_key: str):
        """
        Adds a document to a specific ChromaDB index.
        
        Args:
            document (Document): The document to store.
            index_key (str): Key used to group related documents.
        """
        vectorstore = self._get_vectorstore(index_key)
        vectorstore.add_documents([document])

    def retrieve(self, query: str, index_key: str, k: int = 5):
        """
        Retrieves relevant documents from a specific ChromaDB index based on the query.
        
        Args:
            query (str): Search query.
            index_key (str): Key specifying which index to search.
            k (int): Number of documents to retrieve.
        
        Returns:
            List[Document]: Retrieved documents.
        """
        vectorstore = self._get_vectorstore(index_key)
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        return retriever.get_relevant_documents(query)
