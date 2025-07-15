from src.vector_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.custom_exception import CustomException
from utils.logger import get_logger

logger = get_logger(__name__)


class AnimeRecommendationPipeline:
    def __init__(self, persist_directory: str = "chroma_db"):
        try:
            logger.info("Intializing Recommendation Pipeline")
            vector_build = VectorStoreBuilder(
                csv_path="", persist_directory=persist_directory
            )
            retriever = vector_build.load_vector_store().as_retriever()
            self.recommender = AnimeRecommender(
                retriever=retriever, api_key=GROQ_API_KEY, model_name=MODEL_NAME
            )
            logger.info("Pipeline Intialized Succesffully")
        except Exception as e:
            logger.error(f"Failed to intialize pipeline as - {str(e)}")
            raise CustomException("Error during pipeline initialisation", e)

    def recommend(self, query: str) -> str:
        try:
            logger.info(f"Recieved user query {query}")
            recommendation = self.recommender.get_recommendation(query=query)
            logger.info("Recommendation Generated Succesfully")
            return recommendation
        except Exception as e:
            logger.error(f"Failed to get recommendation as - {str(e)}")
            raise CustomException("Error during recommendation", e)
