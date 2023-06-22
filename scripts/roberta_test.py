import os
import re
import json
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

print(f"Using {DEVICE} device")


roberta_trained_model_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'model', 'roberta_trained_model'))

model = SentenceTransformer(roberta_trained_model_path)

model.to(DEVICE)

animes_file_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'dataset', 'animes_with_cover.json'))

with open(animes_file_path, 'r', encoding='utf-8') as f:
    anime_data = json.load(f)

titles = anime_data['names']  # array com os títulos dos textos
synopses = anime_data['content'][:]  # array com os textos


# search_phrases = ["the soldiers fight to protect the goddess athena",
#                   "the protagonist is a demon who wants to become a hero",
#                   "the protagonist gains the power to kill anyone whose name he writes in a notebook",
#                   "a boy was possessed by a demon and now he has to fight demons",
#                   "the volleyball team is the main focus of the anime",
#                   "the anime shows the daily life of a volleyball team in high school",
#                   "a man who can defeat any enemy with one punch",
#                   "the protagonist become skinny just training",
#                   "it has a dragon wich give three wishes to the one who find it",
#                   "the protagonist always use the wishes to revive his friends",
#                   "the philosopher stone grants immortality to the one who find it",
#                   "two brothers lost their bodies and now they have to find the philosopher stone",
#                   "a ninja kid who wants to become a hokage",
#                   "the protagonist's dream is to become the pirate king",
#                   "the protagonist uses a straw hat and can stretch his body",
#                   "the protagonist got the shinigami sword and now he has to kill hollows",
#                   "it was a knight who use a saint armor blessed by the goddess athena",
#                   "the protagonist met a shinigami and goes to the soul society"]

search_phrases = ["the protagonist gains the power to kill anyone whose name he writes in a notebook",
                  "a man who can defeat any enemy with one punch",
                  "the anime shows a volleyball team which trains to become the best of japan",
                  "the protagonist has the power of stretch his body and use a straw hat",
                  "the sayan warrior revive his friends using the wish given by the dragon",
                  "the philosopher stone grants power and immortality to the one who find it",
                  "two brothers lost their bodies and now they have to find the philosopher stone",
                  "a ninja kid who wants to become the best ninja of his village and has a demon inside him",
                  "the protagonist got the shinigami powers and now he has to kill hollows",
                  "it was a knight who use a saint armor blessed by the goddess athena"]


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


def find_similar(query_text, sinopses_embeddings, k=10):
    '''
        Find the k most similar texts to the query text
    '''
    query_embedding = model.encode([simple_preprocess_text(query_text)])

    similarity_matrix = cosine_similarity(
        query_embedding, sinopses_embeddings).flatten()

    most_similar_indexes = similarity_matrix.argsort()[
        ::-1][:int(k)]

    ranking = []

    for index in most_similar_indexes:
        title = titles[index]

        ranking.append([title, similarity_matrix[index]])

    print("ranking: ", ranking)
    return ranking


def main():
    '''
        Main function
    '''
    cleaned_synopses = [simple_preprocess_text(synopsis)
                        for synopsis in synopses]

    sinopses_embeddings = model.encode(
        cleaned_synopses, show_progress_bar=True)

    print("sinopses_embeddings shape: ", sinopses_embeddings.shape)

    lines = []
    for search_text in search_phrases:
        print("search phrase: ", search_text, "\n")

        results = find_similar(search_text, sinopses_embeddings, k=10)

        lines.append(f"'{search_text}' -->\n")
        lines.append(f"{results}\n\n")

    out_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'public', 'STS_roberta_15k.txt'))

    with open(out_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line)


main()
