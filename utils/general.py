import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm

from utils.plots import classifcation_report


def loss_func(logits, targets, mask, num_labels, entity=False):
    criterion = nn.CrossEntropyLoss()
    if entity:
        active_loss = mask.view(-1) == 1
        active_targets = torch.where(
            active_loss,
            targets.view(-1),
            torch.tensor(criterion.ignore_index).type_as(targets)
        )
        logits = logits.view(-1, num_labels)
        loss = criterion(logits, active_targets)
    else:
        loss = criterion(logits, targets.view(-1))
    return loss


def to_yhat(logits):
    logits = logits.view(-1, logits.shape[-1])
    probs = torch.softmax(logits, dim=1)
    y_hat = torch.argmax(probs, dim=1).cpu().detach().numpy()
    return y_hat


def train_fn(data_loader, model, optimizer, scheduler, device, n_examples, batch=None):
    model = model.train()
    final_loss = 0
    correct_predictions_entity = 0
    correct_predictions_intent = 0
    correct_predictions_scenario = 0
    total_words = 0
    for batch in tqdm(data_loader):
        for k, v in batch.items():
            batch[k] = v.to(device)

        optimizer.zero_grad()

        (entity_logits, intent_logits, scenario_logits) = model(
            batch['ids'], batch['mask'], batch['token_type_ids'])

        entity_loss = loss_func(
            entity_logits, batch['target_entity'], batch['mask'], model.num_entity, entity=True)
        intent_loss = loss_func(
            intent_logits, batch['target_intent'], batch['mask'], model.num_intent)
        scenario_loss = loss_func(
            scenario_logits, batch['target_scenario'], batch['mask'], model.num_scenario)

        loss = (entity_loss + intent_loss + scenario_loss)/3
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_words += (entity_logits.shape[0] * entity_logits.shape[1])
        _, preds_entity = torch.max(entity_logits, dim=-1)
        _, preds_intent = torch.max(intent_logits, dim=-1)
        _, preds_scenario = torch.max(scenario_logits, dim=-1)

        correct_predictions_entity += torch.sum(
            preds_entity.view(-1) == batch['target_entity'].view(-1))
        correct_predictions_intent += torch.sum(
            preds_intent == batch['target_intent'].view(-1))
        correct_predictions_scenario += torch.sum(
            preds_scenario == batch['target_scenario'].view(-1))
        final_loss += loss.item()

    train_loss = final_loss/len(data_loader)
    train_entity_acc = correct_predictions_entity.double()/total_words
    train_intent_acc = correct_predictions_intent.double()/n_examples
    train_scenario_acc = correct_predictions_scenario.double()/n_examples

    return train_loss, train_entity_acc, train_intent_acc, train_scenario_acc


def eval_fn(data_loader, model, device, n_examples, batch=None):
    model = model.eval()
    total_words = 0
    correct_predictions_entity = 0
    correct_predictions_intent = 0
    correct_predictions_scenario = 0
    final_loss = 0

    with torch.no_grad():
        for batch in tqdm(data_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            (entity_logits, intent_logits, scenario_logits) = model(
                batch['ids'], batch['mask'], batch['token_type_ids'])

            entity_loss = loss_func(
                entity_logits, batch['target_entity'], batch['mask'], model.num_entity, entity=True)
            intent_loss = loss_func(
                intent_logits, batch['target_intent'], batch['mask'], model.num_intent)
            scenario_loss = loss_func(
                scenario_logits, batch['target_scenario'], batch['mask'], model.num_scenario)

            loss = (entity_loss + intent_loss + scenario_loss)/3
            final_loss += loss

            total_words += (entity_logits.shape[0] * entity_logits.shape[1])
            _, preds_entity = torch.max(entity_logits, dim=-1)
            _, preds_intent = torch.max(intent_logits, dim=1)
            _, preds_scenario = torch.max(scenario_logits, dim=1)

            correct_predictions_entity += torch.sum(
                preds_entity.view(-1) == batch['target_entity'].view(-1))
            correct_predictions_intent += torch.sum(
                preds_intent == batch['target_intent'].view(-1))
            correct_predictions_scenario += torch.sum(
                preds_scenario == batch['target_scenario'].view(-1))

    val_loss = final_loss/len(data_loader)
    val_entity_acc = correct_predictions_entity.double()/total_words
    val_intent_acc = correct_predictions_intent.double()/n_examples
    val_scenario_acc = correct_predictions_scenario.double()/n_examples
    return val_loss, val_entity_acc, val_intent_acc, val_scenario_acc


def test_fn(data_loader, model, device, enc_list):
    model.eval()
    final_loss = 0
    tasks_y_hats = [None, None, None]
    tasks_targets = [None, None, None]

    precision_dict, recall_dict, fs_dict, fig_dict = {}, {}, {}, {}

    with torch.no_grad():
        for batch in tqdm(data_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)

            (entity_logits, intent_logits, scenario_logits) = model(
                batch['ids'], batch['mask'], batch['token_type_ids'])

            entity_loss = loss_func(
                entity_logits, batch['target_entity'], batch['mask'], model.num_entity, entity=True)
            intent_loss = loss_func(
                intent_logits, batch['target_intent'], batch['mask'], model.num_intent)
            scenario_loss = loss_func(
                scenario_logits, batch['target_scenario'], batch['mask'], model.num_scenario)

            loss = (entity_loss + intent_loss + scenario_loss)/3
            final_loss += loss

            targets_keys = ['target_entity',
                            'target_intent', 'target_scenario']
            logits_list = [entity_logits, intent_logits, scenario_logits]

            for i, (y_hats, target, target_key, logits) in enumerate(zip(tasks_y_hats, tasks_targets, targets_keys, logits_list)):
                if not (y_hats is None):
                    tasks_y_hats[i] = np.concatenate((y_hats, to_yhat(logits)))
                    tasks_targets[i] = np.concatenate(
                        (target, batch[target_key].cpu()))
                else:
                    tasks_y_hats[i] = to_yhat(logits)
                    tasks_targets[i] = batch[target_key].cpu()

        for y_hats, target, enc, key in zip(tasks_y_hats, tasks_targets, enc_list, ['entity', 'intent', 'scenario']):
            precision, recall, fs, fig = classifcation_report(
                y_hats, target, enc)

            precision_dict[key] = precision
            recall_dict[key] = recall
            fs_dict[key] = fs
            fig_dict[key] = fig

    return final_loss/len(data_loader), precision_dict, recall_dict, fs_dict, fig_dict
