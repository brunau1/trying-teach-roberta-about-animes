import os
import random
import re
import json
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from sentence_transformers.evaluation import BinaryClassificationEvaluator


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

print(f"Using {DEVICE} device")

roberta_trained_model_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'model', 'roberta_trained_model'))

roberta_trained_model_checkpoint_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'public', 'checkpoint', 'roberta_trained_model_checkpoint'))

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
    MODEL_NAME, max_seq_length=160)

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
    texts = []
    queries = []
    labels = []

    for _, data in enumerate(train_data['documentRelations']):
        text = synopses[data[1]]
        text = simple_preprocess_text(text)
        query = simple_preprocess_text(data[0])
        # convert the label to float
        label = float(data[2])

        # add the text and query to the arrays
        texts.append(text)
        queries.append(query)
        labels.append(label)

        train_examples.append(
            InputExample(texts=[query, text], label=label))
        # also add the inverse pair
        train_examples.append(
            InputExample(texts=[text, query], label=label))

    print("train examples len: ", len(train_examples))
    print("train example: ", train_examples[0])

    return train_examples, texts, queries, labels


def evaluate(test_samples, model_path):
    '''
        Evaluate the model
    '''
    sts_model = SentenceTransformer(model_path)
    test_evaluator = BinaryClassificationEvaluator.from_input_examples(
        test_samples, name='roberta-test', write_csv=True)
    test_evaluator(sts_model, output_path=model_path)


def train():
    '''
        Train the model
    '''
    train_examples = generate_train_data()[0]

    # randomize the train data
    random.shuffle(train_examples)

    train_samples = []
    dev_samples = []
    test_samples = []

    # Split the train data into train, dev and test
    dev_len = int(len(train_examples) * 0.1)
    test_len = int(len(train_examples) * 0.2)
    train_len = len(train_examples) - dev_len - test_len

    train_samples = train_examples[:len(train_examples)]
    dev_samples = train_examples[train_len:train_len + dev_len]
    test_samples = train_examples[train_len + dev_len:]

    print("train samples len: ", len(train_samples))
    print("dev samples len: ", len(dev_samples))
    print("test samples len: ", len(test_samples))

    # Define train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=24)

    train_loss = losses.MultipleNegativesRankingLoss(model)
    # train_loss = losses.CosineSimilarityLoss(model)

    epochs = 10
    evaluation_steps = 800
    warmup_steps = int(len(train_dataloader) *
                       epochs * 0.1)  # 10% of train data

    print("warmup steps: ", warmup_steps)

    evaluator = BinaryClassificationEvaluator.from_input_examples(
        dev_samples, name='roberta-dev', write_csv=True)

    model.to(DEVICE)
    # Tune the model
    model.fit(train_objectives=[
        (train_dataloader, train_loss)], epochs=epochs, warmup_steps=warmup_steps,
        optimizer_params={'lr': 1e-5}, evaluator=evaluator, evaluation_steps=evaluation_steps,
        output_path=roberta_trained_model_path, save_best_model=True, checkpoint_path=roberta_trained_model_checkpoint_path,
        show_progress_bar=True, checkpoint_save_steps=evaluation_steps)

    # Evaluate the model
    evaluate(test_samples, roberta_trained_model_path)


train()
