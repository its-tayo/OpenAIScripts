# 3. The final result must be a json list of dictionaries with 3 keys: subtopic, description and keywords.
def build_description_generation_prompt(macrotopic, topic, subtopics):
    prompt = f"""The list below are subtopics of the '{topic}' topic and the '{macrotopic}' macrotopic. Your task is to do the following:
1. Provide a comprehensive description of 100 words for each subtopic. Do not include the keywords here
2. Provide at most 5 keywords for each subtopic. The output here must be a string delimited by commas

To enable feeding the output to a topic model, the descriptions could benefit from the following enhancements:
Consistency: Ensure consistency in terminology and language usage across all descriptions to improve the model's ability to identify and extract relevant topics effectively.
Specificity: Provide more specific and detailed information about each subtopic to capture its unique characteristics and distinguish it from other topics.
Diversity: Include a diverse range of keywords and concepts related to each subtopic to enrich the dataset and enable the model to identify a broader spectrum of themes and ideas.
Contextualization: Offer context or examples to illustrate how each subtopic is applied in real-world scenarios or urban contexts, enhancing the model's understanding of the topic's relevance and implications.
Clarity: Ensure clarity and coherence in the descriptions to facilitate accurate interpretation by the model and reduce ambiguity or confusion in topic identification. 
Length: Optimize the length of descriptions to provide a max-100 words information for topic modelling while avoiding excessive verbosity that could overwhelm or dilute the dataset.

Please make sure that the subtopic description includes references to the topic and the macrotopic to maximize the assignment efficacy.

Return your output formatted as a JSON object with the following structure:
{{
"subtopics": [
    {{
        "subtopic": "<SUBTOPIC>",
        "description": "<DESCRIPTION>",
        "keywords" "<KEYWORDS>"
    }}
]
}}

The subtopics are as follows: {subtopics}
"""
    return prompt
