from torch.utils.tensorboard import SummaryWriter
import joblib
from sklearn import model_selection
from torch.utils.data import DataLoader
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np

from utils.processing import process_entity_data, process_intent_scenario_data
import utils.config as config
from utils.general import train_fn, eval_fn, test_fn
from utils.dataset import NLUDataset
from model import NLUModel


def run():
    sentences, target_entity, enc_entity, _ = process_entity_data(
        config.ER_DATASET_PATH)
    target_intent, target_scenario, enc_intent, enc_scenario = process_intent_scenario_data(
        config.IS_DATASET_PATH)

    num_entity, num_intent, num_scenario = len(enc_entity.classes_), len(
        enc_intent.classes_), len(enc_scenario.classes_)

    metadata = {
        'enc_entity': enc_entity,
        'enc_intent': enc_intent,
        'enc_scenario': enc_scenario
    }
    joblib.dump(metadata, 'metadata.bin')

    (train_sentences, test_sentences, train_entity, test_entity, train_intent, test_intent, train_scenario, test_scenario) = model_selection.train_test_split(
        sentences, target_entity, target_intent, target_scenario, random_state=50, test_size=0.1, stratify=target_intent)
    (train_sentences, val_sentences, train_entity, val_entity, train_intent, val_intent, train_scenario, val_scenario) = model_selection.train_test_split(
        train_sentences, train_entity, train_intent, train_scenario, random_state=50, test_size=0.1, stratify=train_intent)

    train_dataset = NLUDataset(
        train_sentences, train_entity, train_intent, train_scenario)
    train_data_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)

    val_dataset = NLUDataset(val_sentences, val_entity,
                             val_intent, val_scenario)
    val_data_loader = DataLoader(
        val_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4)

    test_dataset = NLUDataset(
        test_sentences, test_entity, test_intent, test_scenario)
    test_data_loader = DataLoader(
        test_dataset, batch_size=config.TEST_BATCH_SIZE, num_workers=1)

    device = config.DEVICE
    model = NLUModel(num_entity, num_intent, num_scenario)
    model.to(device)
    if config.MODEL_PATH:
        model.load_state_dict(torch.load(config.MODEL_PATH))

    num_train_steps = config.TRAIN_BATCH_SIZE * config.EPOCHS
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    writer = SummaryWriter(log_dir=config.LOG_PATH)

    best_loss = np.inf
    print(f'Number of Training Sentences:{len(train_sentences)}')
    print(f'Number of Validation Sentences:{len(val_sentences)}')

    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch + 1}/{config.EPOCHS}')
        print('-' * 10)

        (train_loss, train_entity_acc, train_intent_acc, train_scenario_acc) = train_fn(
            train_data_loader, model, optimizer, scheduler, device, len(train_sentences))
        print(f'Train Loss:{train_loss}, Train Entity Acc: {train_entity_acc}, Train Intent Acc: {train_intent_acc}, Train Scenario Acc {train_scenario_acc}')

        (val_loss, val_entity_acc, val_intent_acc, val_scenario_acc) = eval_fn(
            val_data_loader, model, device, len(val_sentences))
        print(
            f'Valid Loss:{val_loss}, Valid Entity Acc: {val_entity_acc} ,Valid Intent Acc: {val_intent_acc}, Train Scenario Acc {val_scenario_acc}')

        if val_loss < best_loss and config.SAVE_MODEL:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = val_loss
            print(f'New Model at Epoch {epoch} , Loss: {best_loss}')

        writer.add_scalars(
            'Loss', {'Train': train_loss, 'Validation': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train Intent': train_intent_acc, 'Train Scenario': train_scenario_acc, 'Train Entity': train_entity_acc,
                           'Valid Intent': val_intent_acc, 'Valid Scenario': val_scenario_acc, 'Valid Entity': val_entity_acc}, epoch)

    enc_list = [enc_entity, enc_intent, enc_scenario]
    run_test(test_data_loader, device, model, enc_list, writer)
    writer.close()


def run_test(test_data_loader, device, model, enc_list, writer):
    test_loss, pre_dict, rec_dict, fs_dict, fig_dict = test_fn(
        test_data_loader, model, device, enc_list)
    writer.add_scalar('Test Loss', test_loss)
    writer.add_scalars('Precision', pre_dict)
    writer.add_scalars('Recall', rec_dict)
    writer.add_scalars('F-score', fs_dict)
    for k, fig in fig_dict.items():
        writer.add_figure(k, fig)


if __name__ == "__main__":
    run()
