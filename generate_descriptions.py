import os
import json
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

# from helpers import count_tokens
from prompts import build_description_generation_prompt


@retry(wait=wait_random_exponential(min=1, max=60 * 3), stop=stop_after_attempt(6))
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("macro_topic")
    parser.add_argument("input_file_path")
    parser.add_argument("output_file_path")
    # parser.add_argument("-ct", "--count_tokens", action="store_true")

    args = parser.parse_args()
    macro_topic = args.macro_topic
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path

    if os.path.isfile(input_file_path) is False:
        print("The target file doesn't exist")
        raise SystemExit(1)

    load_dotenv()

    results = []
    client = OpenAI()
    model = "gpt-3.5-turbo-0125"

    df = pd.read_csv(input_file_path)
    topics = df.columns.to_list()

    for topic in tqdm(topics, desc="topics"):
        subtopics = df[topic].dropna().to_list()
        prompt = build_description_generation_prompt(macro_topic, topic, subtopics)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output JSON that is strictly limited to the format specified in the prompt",
            },
            {"role": "user", "content": prompt},
        ]

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )

        content = json.loads(response.choices[0].message.content)
        result = [
            {"macrotopic": macro_topic, "topic": topic, **item}
            for item in content["subtopics"]
        ]
        results.extend(result)

    results_df = pd.DataFrame(results, columns=list(results[0].keys()))

    directory = os.path.dirname(output_file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main()
