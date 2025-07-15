import streamlit as st

from pipeline.pipeline import AnimeRecommendationPipeline
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.custom_exception import CustomException

st.set_page_config('Anime Recommender', layout="wide")
load_dotenv()

logger =get_logger(__name__)


@st.cache_resource
def init_pipeline():
    return AnimeRecommendationPipeline()


pipeline = init_pipeline()

st.title("Anime Recommender System")

query = st.text_input("Enter your preference Anime")
if query:
    with st.spinner("Fetching recommendation for you ....."):
        try:
            logger.info("Main Pipeline started....")
            response = pipeline.recommend(query=query)
            st.markdown("### Recommendation:")
            st.write(response)
            logger.info(f"The response is shown on Ui for the query  - {query} is - {response}")
        except Exception as e:
            logger.error(f"Failed to show the response on UI - {str(e)}")
            raise CustomException("Failed to display response ", e)

            