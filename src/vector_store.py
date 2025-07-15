from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv

load_dotenv()


class VectorStoreBuilder:
    def __init__(self, csv_path: str, persist_directory: str = "chroma_db"):
        self.csv_path = csv_path
        self.persist_directory = persist_directory
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def load_and_save_vectorstore(self):
        loader = CSVLoader(
            file_path=self.csv_path, encoding="utf-8", metadata_columns=[]
        )
        data = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = splitter.split_documents(data)
        texts = [doc for doc in texts if doc.page_content.strip()]

        db = Chroma.from_texts(
            texts=[doc.page_content for doc in texts],
            embedding=self.embedding,
            persist_directory=self.persist_directory,
        )

        db.persist()

    def load_vector_store(self):
        return Chroma(
            persist_directory=self.persist_directory, embedding_function=self.embedding
        )


if __name__ == "__main__":
    v_store = VectorStoreBuilder(
        csv_path="/home/deependera/LLMOps/anime_recommender/data/anime_processed.csv"
    )
    v_store.load_and_save_vectorstore()
