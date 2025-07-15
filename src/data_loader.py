import pandas as pd


class AnimeDataLoader:
    def __init__(self, original_csv: str, processed_csv: str):
        self.original_csv = original_csv
        self.processed_csv = processed_csv

    def load_and_process(self):
        df = pd.read_csv(
            self.original_csv, encoding="utf-8", on_bad_lines="skip"
        ).dropna()
        required_columns = {"Name", "Genres", "synopsis"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError("Missing Column in CSV file")
        df["combined_info"] = (
            "Title: "+
            df["Name"]+
            ".... Overview: "+
            df["synopsis"]+
            ".... Genres:"+
            df["Genres"]
        )
        df[["combined_info"]].to_csv(self.processed_csv, index=False, encoding="utf-8")
        return self.processed_csv


if __name__ == "__main__":
    neww = AnimeDataLoader(
        "/home/deependera/LLMOps/anime_recommender/data/anime_with_synopsis.csv",
        "/home/deependera/LLMOps/anime_recommender/data/anime_processed.csv",
    )
    neww.load_and_process()
