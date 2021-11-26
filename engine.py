import torch
import joblib
import argparse
from flask import Flask, request, jsonify

import utils.config as config
from utils.inference import entity_extraction, classification, simplify_entities, simplify_intent, simplify_scenario


class NLUEngine:
    def __init__(self, weights):
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
        self.device = config.DEVICE
        self.model = torch.jit.load(weights).to(self.device).eval()

        self.metadata = joblib.load('metadata.bin')
        self.enc_entity = self.metadata['enc_entity']
        self.enc_intent = self.metadata['enc_intent']
        self.enc_scenario = self.metadata['enc_scenario']

        self.num_entity = len(self.enc_entity.classes_)
        self.num_intent = len(self.enc_intent.classes_)
        self.num_scenario = len(self.enc_scenario.classes_)

    def process_sentence(self, text):
        sentence = " ".join(str(text).split())
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len
        )

        tokenized_ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
        word_pieces = self.tokenizer.decode(inputs['input_ids']).split()[1:-1]

        padding_len = self.max_len - len(tokenized_ids)

        ids = tokenized_ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)

        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(
            0).to(self.device)
        token_type_ids = torch.tensor(
            token_type_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        return ids, mask, token_type_ids, tokenized_ids, word_pieces

    def predict_sentence(self, ids, mask, token_type_ids):
        entity_hs, intent_hs, scenario_hs = self.model(
            ids, mask, token_type_ids)
        return entity_hs, intent_hs, scenario_hs

    def predict(self, sentence):
        ids, mask, token_type_ids, tokenized_ids, word_pieces = self.process_sentence(
            sentence)
        entity_hs, intent_hs, scenario_hs = self.predict_sentence(
            ids, mask, token_type_ids)
        words_labels_json, words_scores_json = entity_extraction(
            self.enc_entity, entity_hs, word_pieces, tokenized_ids)

        intent_sentence_labels_json, intent_class_scores_json = classification(
            self.enc_intent, self.enc_scenario, intent_hs, task='intent')
        scenario_sentence_labels_json, scenario_class_scores_json = classification(
            self.enc_intent, self.enc_scenario, scenario_hs, task='scenario')

        prediction = {'sentence': sentence}
        prediction['entities'] = simplify_entities(
            words_labels_json, words_scores_json)
        prediction['intent'] = simplify_intent(
            intent_sentence_labels_json, intent_class_scores_json)
        prediction['scenario'] = simplify_scenario(
            scenario_sentence_labels_json, scenario_class_scores_json)

        return prediction


app = Flask(__name__)
nlu_engine: NLUEngine = None


@app.route("/nlu_engine")
def predict():
    sentence = request.args.get('sentence')
    predictions = nlu_engine.predict(sentence)
    print(predictions)
    return jsonify(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HomeIO NLU Engine")
    parser.add_argument('--weights', type=str, default=config.MODEL_PATH, required=True,
                        help='Optimized Weights for Model. use optimize_weights.py')
    args = parser.parse_args()

    nlu_engine = NLUEngine(args.weights)

    # test_sentence = 'wake me up at 5 am please'
    # prediction = nlu_engine.predict(test_sentence)
    # print(prediction)

    app.run(debug=True)
