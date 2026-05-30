from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from uuid import uuid4
from retriever import Retriever
# Directory Loader


if __name__ == '__main__':
    loader = DirectoryLoader(".\private-docs", glob="*.pdf", loader_cls=PyPDFLoader)
    data = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    documents = text_splitter.split_documents(data)
    r = Retriever()
    r.index_documents(documents)
    print('Seeded docs')
