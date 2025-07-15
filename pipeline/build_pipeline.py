from src.data_loader import AnimeDataLoader
from src.vector_store import VectorStoreBuilder
from utils.logger import get_logger
from utils.custom_exception import CustomException
from dotenv import load_dotenv

load_dotenv()

logger = get_logger(__name__)


def main():
    try:
        logger.info("Start to build the pipeline....")
        loader = AnimeDataLoader(
            original_csv="/home/deependera/LLMOps/anime_recommender/data/anime_with_synopsis.csv",
            processed_csv="/home/deependera/LLMOps/anime_recommender/data/processed_anime.csv",
        )
        processed_csv = loader.load_and_process()
        logger.info("Data loaded and processed")
        vector_builder = VectorStoreBuilder(csv_path=processed_csv)
        vector_builder.load_and_save_vectorstore()
        logger.info("Vector store Build Succesfully...")
        logger.info("Pipeline built successfully")
    except Exception as e:
        logger.error(f"Failed to build pipeline as - {str(e)}")
        raise CustomException("Error during pipeline building", e)


if __name__ == "__main__":
    main()
    print("Run main")