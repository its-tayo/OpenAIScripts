import os
import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("group_column")
    parser.add_argument("column")
    parser.add_argument("input_file_path")

    args = parser.parse_args()

    column = args.column
    group_column = args.group_column
    input_file_path = args.input_file_path

    if os.path.isfile(input_file_path) is False:
        print("The target file doesn't exist")
        raise SystemExit(1)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    output_path = f"./plots/{Path(input_file_path).stem}"
    Path(output_path).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_file_path)
    df = df[[group_column, column]]

    topics = df[group_column].unique().tolist()

    for topic in tqdm(topics, desc="topics"):
        filename = topic.strip().replace(" ", "_")
        descriptions = df[df[group_column] == topic][column].to_list()
        embeddings = model.encode(descriptions)

        cos_sim = util.cos_sim(embeddings, embeddings)

        plt.subplots(figsize=(15, 15))
        sns.heatmap(cos_sim, annot=True, fmt=".2f")
        plt.savefig(f"{output_path}/{filename}.png")


if __name__ == "__main__":
    main()
