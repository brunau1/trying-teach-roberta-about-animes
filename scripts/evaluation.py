import os
import re
import json
import torch
import random
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

train_data_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'dataset', 'roberta-train-sentence-documents-relations_v8.json'))

animes_file_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'dataset', 'animes_with_cover.json'))

with open(animes_file_path, 'r', encoding='utf-8') as f:
    anime_data = json.load(f)

with open(train_data_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)

titles = anime_data['names']  # array com os títulos dos textos
synopses = anime_data['content']  # array com os textos


def calculate_f_score(relevant_docs, retrieved_docs):
    precision = len(set(relevant_docs).intersection(
        set(retrieved_docs))) / float(len(retrieved_docs))
    recall = len(set(relevant_docs).intersection(
        set(retrieved_docs))) / float(len(relevant_docs))
    if precision + recall == 0:
        return 0
    f_score = 2 * (precision * recall) / (precision + recall)
    return f_score


def simple_preprocess_text(text=''):
    '''
        Preprocess the text to remove special characters, line breaks, tabs, double spaces and numbers
    '''
    text = text.lower()
    # remove all special characters
    text = re.sub(r'[^\w\s]', '', text)
    # remove line breaks and tabs
    text = re.sub(r'(?:\n|\r|\t|\s{2})', ' ', text)
    # remove all numbers
    text = re.sub(r'\d+', '', text)
    # remove double spaces
    text = re.sub(r'\s+', ' ', text)

    return text


def find_similar(model, query_text, sinopses_embeddings, k=10):
    '''
        Find the k most similar texts to the query text
    '''
    query_embedding = model.encode([simple_preprocess_text(query_text)])

    similarity_matrix = cosine_similarity(
        query_embedding, sinopses_embeddings).flatten()

    most_similar_indexes = similarity_matrix.argsort()[
        ::-1][:int(k)]

    # return the first result
    return most_similar_indexes[0]


def generate_eval_data():
    # open saved data for evaluation else generate new data
    try:
        eval_data_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'public', 'dataset', 'roberta-eval-data.json'))

        with open(eval_data_path, 'r', encoding='utf-8') as f:
            eval_data = json.load(f)

        return eval_data

    except FileNotFoundError:
        eval_data = []

        for _, data in enumerate(train_data['documentRelations']):
            query = simple_preprocess_text(data[0])
            relevant_doc_id = data[1]

            eval_data.append({
                'query': query,
                'relevant_doc_id': relevant_doc_id
            })

        random.shuffle(eval_data)

        data_for_eval = eval_data[:int(len(eval_data) * 0.1)]

        # save the data for evaluation
        out_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'public', 'dataset', 'roberta-eval-data.json'))

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(data_for_eval, f, indent=4)

        # return 10% of the data
        return data_for_eval


def evaluate(model, eval_data):
    print("evaluate")

    cleaned_synopses = [simple_preprocess_text(synopsis)
                        for synopsis in synopses]

    sinopses_embeddings = model.encode(
        cleaned_synopses, show_progress_bar=True)

    computed_f1 = []

    print("computing f1")

    for _, data in tqdm(enumerate(eval_data)):
        query = data['query']
        relevant_doc_id = data['relevant_doc_id']

        retrieved_doc_id = find_similar(
            model, query, sinopses_embeddings)

        f1 = calculate_f_score([relevant_doc_id], [retrieved_doc_id])

        computed_f1.append(f1)

    print("computed_f1: ", computed_f1)
    print("mean: ", np.mean(computed_f1))

    return np.mean(computed_f1), computed_f1


def main():
    model_number = [
        '26400',
        '28800',
        '29600',
        '30400',
        '31200',
        '31540'
    ]

    eval_data = generate_eval_data()

    print("eval_data len: ", len(eval_data))

    for number in model_number:
        roberta_trained_model_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'public', 'checkpoint', 'roberta_trained_model_checkpoint', number))

        sts_model = SentenceTransformer(roberta_trained_model_path)

        sts_model.to(DEVICE)

        print("model: ", number)

        evaluation = evaluate(sts_model, eval_data)

        out_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'public', f"evaluation_{number}.txt"))

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(f"evaluation: {evaluation[0]}\n")
            f.write(f"computed_f1: {evaluation[1]}\n")


main()
