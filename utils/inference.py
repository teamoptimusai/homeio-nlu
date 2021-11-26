import torch


def classes_to_scores_json(classes, intent_scores):
    dict = {'scores': []}
    for scores in intent_scores:
        dict['scores'] += [{c: s for c, s in zip(classes, scores)}]
    return dict


def sentence_to_labels_json(labels, num_sentence):
    return {i: l for l, i in zip(labels, range(num_sentence))}


def words_to_labels_json(word_pieces, labels):
    return {w: l for w, l in zip(word_pieces, labels)}


def words_to_scores_json(words_pieces, scores):
    return {w: cs for w, cs in zip(words_pieces, scores)}


def to_yhat(logits):
    logits = logits.view(-1, logits.shape[-1]).cpu().detach()
    probs = torch.softmax(logits, dim=1)
    y_hat = torch.argmax(probs, dim=1)
    return probs.numpy(), y_hat.numpy()


def simplify_entities(words_labels, words_scores):
    entities = [{'word': word_label, 'entity': words_labels[word_label], 'score': words_scores[word_label].astype(float)
                 [words_scores[word_label].argmax()]} for word_label in words_labels if (words_labels[word_label] != 'O')]
    return entities


def simplify_intent(intent_sentence_labels, intent_class_scores):
    return {'class': intent_sentence_labels[0], 'score': intent_class_scores['scores'][0][intent_sentence_labels[0]].astype(float)}


def simplify_scenario(scenario_sentence_labels, scenario_class_scores):
    return {'class': scenario_sentence_labels[0], 'score': scenario_class_scores['scores'][0][scenario_sentence_labels[0]].astype(float)}


def entity_extraction(enc_entity, entity_hs, word_pieces, tokenized_ids):
    entity_scores, entity_preds = to_yhat(entity_hs)
    entity_scores = entity_scores[1:len(tokenized_ids)-1, :]
    enitity_labels = enc_entity.inverse_transform(
        entity_preds)[1:len(tokenized_ids)-1]
    words_labels_json = words_to_labels_json(word_pieces, enitity_labels)
    words_scores_json = words_to_scores_json(word_pieces, entity_scores)
    return words_labels_json, words_scores_json


def classification(enc_intent, enc_scenario, logits, task='intent'):
    if task == 'intent':
        enc = enc_intent
    else:
        enc = enc_scenario
    class_scores, class_preds = to_yhat(logits)
    sentence_labels_json = sentence_to_labels_json(
        enc.inverse_transform(class_preds), len(class_preds))
    class_scores_json = classes_to_scores_json(enc.classes_, class_scores)
    return sentence_labels_json, class_scores_json
