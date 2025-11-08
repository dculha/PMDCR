import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import pandas as pd
import time
import json
import ast
import nltk
from nltk.tokenize import sent_tokenize
from openai import OpenAI
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import anthropic
import os
from mistralai import Mistral
import string
import re
from sympy import isprime
from sympy import nextprime
import uuid
import transformers
import torch
from huggingface_hub import InferenceClient
from xai_sdk import Client
from xai_sdk.chat import user, system


FILES_PLACE = "FILES_PLACE"

genai.configure(api_key="API")

llama_tokenizer = None
llama_model = None

def gemini_prompt(prompt, model_name):
    gemini_model = genai.GenerativeModel(model_name)
    gemini_response = gemini_model.generate_content(prompt)

    if not gemini_response.candidates:
        return ""

    candidate = gemini_response.candidates[0]
    if not candidate.content.parts:
        return ""

    return candidate.content.parts[0].text

client = OpenAI(
    api_key="API",
)

anthropic_client = anthropic.Anthropic(api_key='API')

nltk.download('punkt')  
nltk.download('punkt_tab')  

def openai_prompt(prompt, model_name):
    messages=[
        {"role": "user", "content": prompt},
    ]
    response = client.chat.completions.create(model = model_name, messages = messages)
    result = ""
    try:
        result = response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred (openai): {e}")
    return result    

XAI_API = "API"

xai_client = OpenAI(
    api_key=XAI_API,
    base_url="https://api.x.ai/v1"   # xAI base
)

def xai_prompt(prompt, model_name):
    messages=[
        {"role": "user", "content": prompt},
    ]
    response = xai_client.chat.completions.create(model = model_name, messages = messages)
    result = ""
    try:
        result = response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred (xai): {e}")
    return result    

def claude_prompt(prompt, model_name):
    response = anthropic_client.messages.create(
        model = model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    result = ""
    try:
        result = response.content[0].text
    except Exception as e:
        print(f"An error occurred (claude): {e}")
    return result

def huggingface_prompt(prompt, model_name):
    client = InferenceClient(model_name, token="TOKEN")
    messages = [{"role": "user", "content": prompt}]
    response = client.chat_completion(messages=messages)
    try:
        result = response.choices[0].message["content"]
    except Exception as e:
        print(f"An error occurred (claude): {e}")
    return result

def make_request_with_retry(request_func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return request_func()
        except Exception as e:
            print(f"Retry {attempt + 1}/{max_retries} due to {e}")
            if str(e).startswith("Invalid operation"):
                raise Exception("Invalid operation")
            time.sleep(2 ** attempt)  # Exponential backoff
    raise Exception("Max retries exceeded")

def randomize_sentences(df, index, row, text_column, randomized_column):
    value = df.at[index, randomized_column]
    if pd.isna(value):
        text = value = df.at[index, text_column]
        sentences = sent_tokenize(text)
        sentences_df = pd.DataFrame(sentences, columns=["Name"])
        sentences_df['Randomized'] = sentences_df['Name'].sample(frac=1).reset_index(drop=True)
        continuous_text = sentences_df['Randomized'].str.cat(sep=' ')
        df.at[index, randomized_column] = continuous_text
        df.to_csv(input_file_name, index=False)

def randomize_sentence_words(sentence):
    words = sentence.split()  # Split the sentence into words
    random.shuffle(words)  # Shuffle the words randomly
    return ' '.join(words)  # Join the shuffled words back into a string

def randomize_sentences_words(df, index, row, text_column, randomized_column):
    value = df.at[index, randomized_column]
    if pd.isna(value):
        text = value = df.at[index, text_column]
        sentences = sent_tokenize(text)
        sentences_words = []
        for sentence in sentences:
            randomized_sentence = randomize_sentence_words(sentence)
            sentences_words.append(randomized_sentence)
        sentences_df = pd.DataFrame(sentences_words, columns=["Name"])
        sentences_df['Randomized'] = sentences_df['Name'].sample(frac=1).reset_index(drop=True)
        continuous_text = sentences_df['Randomized'].str.cat(sep=' ')
        df.at[index, randomized_column] = continuous_text
        df.to_csv(input_file_name, index=False)

def randomize_sentences_words_2(df, index, row, text_column, randomized_column):
    value = df.at[index, randomized_column]
    if pd.isna(value):
        text = value = df.at[index, text_column]
        sentences = sent_tokenize(text)
        length = len(sentences)
        sentences_words = []
        for sentence in sentences:
            randomized_sentence = randomize_sentence_words(sentence)
            sentences_words.append(randomized_sentence)
        sentences_df = pd.DataFrame(sentences_words, columns=["Name"])
        sentences_df['Randomized'] = sentences_df['Name'].sample(frac=1).reset_index(drop=True)

        cutoff = int(len(sentences_df) * 0.5) #remove half of the sentences
        sentences_df = sentences_df.iloc[cutoff:].reset_index(drop=True) #remove first row

        continuous_text = sentences_df['Randomized'].str.cat(sep=' ')
        df.at[index, randomized_column] = continuous_text
        df.to_csv(input_file_name, index=False)

def randomize_into_two_sentences_words(df, index, row, text_column, randomized_column1, randomized_column2):
    value1 = df.at[index, randomized_column1]
    value2 = df.at[index, randomized_column2]
    if pd.isna(value1) or pd.isna(value2):
        text = df.at[index, text_column]
        sentences = sent_tokenize(text)
        sentences_words = []
        for sentence in sentences:
            randomized_sentence = randomize_sentence_words(sentence)
            sentences_words.append(randomized_sentence)
        sentences_df = pd.DataFrame(sentences_words, columns=["Name"])
        sentences_df['Randomized'] = sentences_df['Name'].sample(frac=1).reset_index(drop=True)

        cutoff = int(len(sentences_df) * 0.5) 

        sentences1_df = sentences_df.iloc[0:cutoff].reset_index(drop=True) 
        sentences2_df = sentences_df.iloc[cutoff:].reset_index(drop=True) 

        continuous_text = sentences1_df['Randomized'].str.cat(sep=' ')
        df.at[index, randomized_column1] = continuous_text

        continuous_text = sentences2_df['Randomized'].str.cat(sep=' ')
        df.at[index, randomized_column2] = continuous_text

        df.to_csv(input_file_name, index=False)

def find_similarity_of_texts(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate cosine similarity
    cos_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return cos_sim[0][0]

def find_similarity(df, index, row, first_column, second_column, equality_column):
    try:
        value = df.at[index, equality_column]
        if pd.isna(value):
            first_response = df.at[index, first_column]
            second_response = df.at[index, second_column]
            df.at[index, equality_column] = find_similarity_of_texts(first_response, second_response)

            df.to_csv(input_file_name, index=False)
    except Exception as e:
        print(f"An error occurred: {e}")

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Function to calculate ROUGE-L score
def calculate_rouge_l(reference, generated):
    scores = scorer.score(reference, generated)
    return scores['rougeL'].fmeasure  

def find_equality(df, index, row, first_column, second_column, equality_column):
    try:
        value = df.at[index, equality_column]
        if pd.isna(value):
            first_response = df.at[index, first_column]
            second_response = df.at[index, second_column]
            df.at[index, equality_column] = calculate_rouge_l(first_response, second_response)

            df.to_csv(input_file_name, index=False)
    except Exception as e:
        print(f"An error occurred: {e}")

def compare_equality(df, index, row, first_column, second_column, equality_column, equality_progress_column):
    try:
        value = df.at[index, equality_column]
        if pd.isna(value):
            first_response = df.at[index, first_column]
            second_response = df.at[index, second_column]
            equality = 0
            try:
                if int(first_response) == int(second_response):
                    equality = 1
            except Exception as e:
                print(f"An error occurred: {e}") 
            df.at[index, equality_column] = equality

            sum_equality = df[equality_column].iloc[:index+1].sum()
            sum_equality = sum_equality * 100 / int(index+1)
            df.at[index, equality_progress_column] = sum_equality

            print("sum_equality", sum_equality)            

            df.to_csv(input_file_name, index=False)     

            if sum_equality < 50:
                print("VIOLATION: sum_equality < 50")
                exit()    
    except Exception as e:
        print(f"An error occurred: {e}")    

def prepare_prompt1(df, index):
    text = df.at[index, "prompt"]
    prompt = text
    return prompt

def prepare_prompt2(df, index):
    randomized_text1 = df.at[index, "randomized_sentences1"]
    randomized_text2 = df.at[index, "randomized_sentences2"]
    prompt = "The first text is:\n" + randomized_text1 + "\nThe second text is:\n" + randomized_text2 \
        + "\nMerge these two texts, return only the output:\n"
    return prompt

def prepare_prompt3(df, index):
    sco = df.at[index, "sco_response"]
    randomized_text1 = df.at[index, "randomized_sentences1"]
    randomized_text2 = df.at[index, "randomized_sentences2"]
    prompt = "The first text is:\n" + randomized_text1 + "\nThe second text is:\n" + randomized_text2 \
        + "\nThe structured content outline is:\n" + sco \
        + "\nDo not include any part of the given structured content outline, merge the first and second texts by the help of the given structured content outline, return only the merged texts:\n" + sco
    return prompt

def is_level_completed(df, model_name, level):
    filtered = df[(df["model_name"] == model_name) & (df["level_end"] != "end")]    
    last_duplicate = filtered.duplicated(subset='normalized_response', keep="first")
    df.loc[filtered.index[last_duplicate], "level_end"] = "end"
    df.to_csv(input_file_name, index=False)

    filtered = df[(df["model_name"] == model_name) & (df["level"] == level)]
    if filtered.empty:
        return False
    else:
        filtered = filtered[(filtered["level_end"] == "end")]    
        if filtered.empty:
            return False
        else:
            return True


def remove_model_records(model_name):
    global df
    condition = (df["model_name"] == model_name)
    df = df[~condition]
    df.to_csv(input_file_name, index=False)

def determine_random_concept(df, model_name, level):
    if level == 1:
        return ""
    else:
        df_selected = df[(df["model_name"] == model_name) & (df["level"] == (level - 1))]
        if len(df_selected) == 0:
            return ""
        df_selected = df_selected.iloc[:-1]
        if len(df_selected) == 0:
            return ""
        random_row = df_selected.sample(n=1, random_state=42)
        return random_row["normalized_response"].values[0]

def normalize_response(response):
    result = response.lower()
    result = re.sub(rf'[\n\r\t{re.escape(string.punctuation)}]', '', result)
    #result = result[:100]
    return result

def is_prime(position):
    n = int(position)
    return isprime(n)

def execute_prompt(model_selection, position, prompt, power, opposed_to_part, fundamental_concept):
    global df
    start_time = time.time()   
    
    model_name = model_selection.get_model_name()
    model_execution_function = model_selection.get_model_execution_function(prompt)

    response = make_request_with_retry(model_execution_function, 1000)
    
    end_time = time.time()

    normalized_response = normalize_response(response)
    if opposed_to_part == "":
        opposed_to_part = normalized_response
    else:
        opposed_to_part += ", " + normalized_response

    time_elapsep = end_time - start_time    

    correct = 1
    if normalized_response == "no":
        normalized_response = fundamental_concept
        correct = 0

    (num_distincts, num_repetitions)  = find_num_distincts_in_model(df, model_name)

    count = find_num_rows_in_model(df, model_name)
    count += 1
    sum = find_sum_in_model(df, model_name)
    sum += correct
    success = (sum * 100)/count
    new_row_values = {
        "model_name": model_name,
        "position": position,
        "prompt": prompt,
        "response": response,
        "normalized_response": normalized_response,
        "time_elapsed": time_elapsep,
        "distincts": num_distincts,
        "repetitions": num_repetitions,
        "correct": correct,
        "success": success,
        "power": power,
        "opposed_to_part": opposed_to_part
    }
    new_row = pd.DataFrame([new_row_values])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(input_file_name, index=False)

    return (normalized_response, num_distincts, num_repetitions, success, correct, opposed_to_part)

def prepare_concept_prompt(opposed_to_part):
    if opposed_to_part == "":
        prompt = "What is the most important fundamental concept? Just return the concept."
    else:
        prompt = "What is the concept or the compound concept that is most opposed to the following concepts: " + opposed_to_part + "? If your response is a phrase or a sentence, just return no; if the response is a concept or a compound concept , just return the response; otherwise just return no."        
    return prompt

def find_num_rows_in_model(df, model_name):
    df_selected = df[(df["model_name"] == model_name)]
    count = len(df_selected)
    return count

def find_sum_in_model(df, model_name):
    df_selected = df[(df["model_name"] == model_name)]
    sum = df_selected["correct"].sum()
    return sum

def find_num_distincts_in_model(df, model_name):
    df_selected = df[(df["model_name"] == model_name)]
    if df_selected.empty:
        return (0, 0)
    else:
        filtered = df[(df["model_name"] == model_name)]    
        num_distincts = filtered['normalized_response'].nunique()
        num_rows = filtered['normalized_response'].count()
        num_repetitions = num_rows - num_distincts
        return (num_distincts, num_repetitions)
    
def find_next_position_previous_opposed_to_part(df, model_name):
    df_selected = df[(df["model_name"] == model_name)]
    if df_selected.empty:
        return 1, ""
    else:
        filtered = df[(df["model_name"] == model_name)]    
        max = filtered['position'].max()
        filtered_max = filtered[(filtered["position"] == max)]
        opposed_to_part = filtered_max["opposed_to_part"].iloc[0]
        max += 1
        return max, opposed_to_part
    
def find_fundamental_concept(df, model_name):
    df_selected = df[(df["model_name"] == model_name)]
    if df_selected.empty:
        return ""
    else:
        filtered = df[(df["model_name"] == model_name)]    
        min = filtered['position'].min()
        filtered_min = filtered[(filtered["position"] == min)]
        fundamental_concept = filtered_min["normalized_response"].iloc[0]
        return fundamental_concept
    
def find_next_power_for_model(df, model_name):
    df_selected = df[(df["model_name"] == model_name)]
    if df_selected.empty:
        return 4
    else:
        filtered = df[(df["model_name"] == model_name)]    
        max = filtered['power'].max()
        max += 1
        return int(max)

def find_max_position_min_success_in_model(df, model_name):
    df_selected = df[(df["model_name"] == model_name)]
    if df_selected.empty:
        return "", 100
    else:
        filtered = df[(df["model_name"] == model_name)]    
        max_position = filtered['position'].max()
        min_success = filtered['success'].min()
        return max_position, min_success
    
def mersenne_number(power):
    m = 2 ** power - 1
    return m
    
def find_next_position(power):
    if power % 2 == 1:
        half = power // 2
        half_number = mersenne_number(half)
        half_prime = nextprime(half_number)
        bigger = half + 1
        bigger_number = mersenne_number(bigger)
        bigger_prime = nextprime(bigger_number)
        result = half_prime * bigger_prime
        return str(result), power + 1
    else:
        number = mersenne_number(power)
        prime = nextprime(number)
        return prime, power + 1

class ModelFamily:
    def __init__(self, model_family, models):
        self.model_family = model_family
        self.models = models

    def get_models(self):
        return self.models
    
    def get_family_name(self):
        return self.model_family

class ModelSelection:
    def __init__(self):
        self.model_families = []
        
        models = [
            "gemini-1.5-flash-8b",
            "gemini-1.5-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            "gemini-2.5-flash-preview-05-20"
        ]
        model_family = ModelFamily("gemini", models)
        #self.model_families.append(model_family)

        models = [
            "gpt-3.5-turbo-0125",
            "gpt-4-0613",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-08-06",
            "gpt-4.5-preview-2025-02-27",
            "gpt-4.1-2025-04-14"
        ]
        model_family = ModelFamily("openai", models)
        #self.model_families.append(model_family)   

        models = [
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-7-sonnet-20250219",
            "claude-sonnet-4-20250514",
            "claude-opus-4-20250514"
        ]
        model_family = ModelFamily("claude", models)
        #self.model_families.append(model_family)   

        models = [
            "meta-llama/Llama-3.3-70B-Instruct"
        ]
        model_family = ModelFamily("llama", models)
        #self.model_families.append(model_family)  
        
        models = [
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
        model_family = ModelFamily("mistral", models)
        #self.model_families.append(model_family)  

        models = [
            "google/gemma-3-12b-it"
        ]
        model_family = ModelFamily("gemma", models)
        #self.model_families.append(model_family) 

        models = [
            "grok-3"
        ]
        model_family = ModelFamily("xai", models)
        self.model_families.append(model_family) 

        self.model_family_index = 0
        self.model_family_iteration = 0

    def iterate(self):
        while(True):
            if self.model_family_index < len(self.model_families):
                model_family = self.model_families[self.model_family_index]
                if self.model_family_iteration < len(model_family.get_models()):
                    self.model_family_iteration += 1
                    if self.model_family_iteration < len(model_family.get_models()):
                        break
                    else:
                        self.model_family_index += 1
                        self.model_family_iteration = -1
                else:
                    self.model_family_index += 1
                    self.model_family_iteration = -1
            else:
                return False
        return True

    def get_model_name(self):
        model_name = self.model_families[self.model_family_index].get_models()[self.model_family_iteration]
        return model_name
            
    def get_model_execution_function(self, prompt):
        model_family_name = self.model_families[self.model_family_index].get_family_name()
        if model_family_name == "gemini":
            return lambda: gemini_prompt(prompt, self.get_model_name())
        elif model_family_name == "openai":
            return lambda: openai_prompt(prompt, self.get_model_name())
        elif model_family_name == "claude":
            return lambda: claude_prompt(prompt, self.get_model_name())
        elif model_family_name == "llama":
            return lambda: huggingface_prompt(prompt, self.get_model_name())        
        elif model_family_name == "mistral":
            return lambda: huggingface_prompt(prompt, self.get_model_name())        
        elif model_family_name == "gemma":
            return lambda: huggingface_prompt(prompt, self.get_model_name())       
        elif model_family_name == "xai":
            return lambda: xai_prompt(prompt, self.get_model_name())              
        else:
            raise "ERROR"

#program starts

model_selection = ModelSelection()

columns_array = [
    "model_name",    
    "position",
    "prompt",
    "response",
    "normalized_response",
    "time_elapsed",
    "distincts",
    "repetitions",
    "correct",
    "success"
    ]


input_file_name = FILES_PLACE + 'dataset.csv' 

if not os.path.exists(input_file_name):
    df = pd.DataFrame(columns=columns_array)
    df.to_csv(input_file_name, index=False, encoding='latin1')
    print(f"File created: {input_file_name}")
else:
    print(f"File found: {input_file_name}")

df = pd.read_csv(input_file_name, encoding='latin1')

while(True):
    model_name = model_selection.get_model_name()

    position, opposed_to_part = find_next_position_previous_opposed_to_part(df, model_name)
    previous_position = position - 1

    correct = 1
    response = ""

    success = 100
    fundamental_concept = ""
    while True:
        power = position

        (num_distincts, num_repetitions) = find_num_distincts_in_model(df, model_name)
        print(model_name, "#repetitions:", num_repetitions, "#distincts:", num_distincts, "success:", success, "power:", power )
        if num_repetitions >= 5:
            break     
        if fundamental_concept == "":
            fundamental_concept = find_fundamental_concept(df, model_name)   
        print("###Position", previous_position, ":", response)
        prompt = prepare_concept_prompt(opposed_to_part)
        (response, num_distincts, num_repetitions, success, correct, opposed_to_part) = execute_prompt(model_selection, position, prompt, power, opposed_to_part, fundamental_concept)
        previous_position = position    
        position += 1

    if not model_selection.iterate():
        break

    print()



