from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_groq import ChatGroq
from src.prompt_template import get_anime_prompt

from config.config import GROQ_API_KEY, MODEL_NAME

from src.vector_store import VectorStoreBuilder

from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq
# from src.prompt_template import get_anime_prompt

class AnimeRecommender:
    def __init__(self,retriever,api_key:str,model_name:str):
        self.llm = ChatGroq(api_key=api_key,model=model_name,temperature=0)
        self.prompt = get_anime_prompt()

        self.qa_chain = RetrievalQA.from_chain_type(
            llm = self.llm,
            chain_type = "stuff",
            retriever = retriever,
            return_source_documents = True,
            chain_type_kwargs = {"prompt":self.prompt}
        )

    def get_recommendation(self,query:str):
        result = self.qa_chain.invoke({"query":query})
        return result['result']


if __name__ =="__main__":

    vector_store = VectorStoreBuilder(csv_path="/home/deependera/LLMOps/anime_recommender/data/anime_processed.csv")
    chroma = vector_store.load_vector_store()
    retriever = chroma.as_retriever()
    anime = AnimeRecommender(retriever=retriever,api_key=GROQ_API_KEY, model_name=MODEL_NAME)
    result = anime.get_recommendation(query="What is Anime?")
    print(result)