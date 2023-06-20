import os
import re
import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

print(f"Using {DEVICE} device")

roberta_trained_model_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'model', 'roberta_trained_model'))

animes_file_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'dataset', 'animes_with_cover.json'))

train_data_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'dataset', 'roberta-train-sentence-documents-relations_v8.json'))

with open(animes_file_path, 'r', encoding='utf-8') as f:
    anime_data = json.load(f)

with open(train_data_path, 'r', encoding='utf-8') as f:
    train_data = json.load(f)


titles = anime_data['names']  # array com os títulos dos textos
synopses = anime_data['content']  # array com os textos

# distil roberta is used because it is smaller and faster
MODEL_NAME = 'distilroberta-base'

word_embedding_model = models.Transformer(
    MODEL_NAME, max_seq_length=256)

pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


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


def generate_train_data():
    '''
        Generate the train data for the model
    '''

    train_examples = []

    for _, data in enumerate(train_data['documentRelations']):
        text = synopses[data[1]]
        text = simple_preprocess_text(text)
        query = simple_preprocess_text(data[0])
        # convert the label to float
        label = float(data[2])

        train_examples.append(
            InputExample(texts=[query, text], label=label))
        # also add the inverse pair
        train_examples.append(
            InputExample(texts=[text, query], label=label))

    print("train examples len: ", len(train_examples))
    print("train example: ", train_examples[0])

    return train_examples


def train():
    '''
        Train the model
    '''
    train_examples = generate_train_data()

    # Define train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=24)

    train_loss = losses.MultipleNegativesRankingLoss(model)
    # train_loss = losses.CosineSimilarityLoss(model)

    epochs = 4

    warmup_steps = int(len(train_dataloader) *
                       epochs * 0.1)  # 10% of train data

    model.to(DEVICE)
    # Tune the model
    model.fit(train_objectives=[
        (train_dataloader, train_loss)], epochs=epochs, warmup_steps=warmup_steps, optimizer_params={'lr': 1e-5})

    # Save the model
    model.save(roberta_trained_model_path)


train()
