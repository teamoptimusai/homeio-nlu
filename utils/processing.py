from sklearn import preprocessing
import pandas as pd


def process_entity_data(data_path):
    df = pd.read_csv(data_path, encoding='latin-1')

    enc_entity = preprocessing.LabelEncoder()

    df.loc[:, 'entity'] = enc_entity.fit_transform(df['entity'])

    sentences = df.groupby('Sentence #')['words'].apply(list).values
    entity = df.groupby('Sentence #')['entity'].apply(list).values
    len_upper = len(max(sentences, key=len))
    return sentences, entity, enc_entity, len_upper


def process_intent_scenario_data(data_path):
    df = pd.read_csv(data_path, encoding='latin-1')

    enc_intent = preprocessing.LabelEncoder()
    df.loc[:, 'intent'] = enc_intent.fit_transform(df['intent'])

    enc_scenario = preprocessing.LabelEncoder()
    df.loc[:, 'scenario'] = enc_scenario.fit_transform(df['scenario'])

    intent = df.groupby('Sentence #')['intent'].apply(list).values
    scenario = df.groupby('Sentence #')['scenario'].apply(list).values

    return intent, scenario, enc_intent, enc_scenario
