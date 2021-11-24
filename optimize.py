import torch
import joblib
import argparse

from model import NLUModel
import utils.config as config


def trace(model):
    model.eval()
    ids = torch.zeros(1, 61, dtype=torch.long)
    cls = torch.tensor([[101]], dtype=torch.long)
    sep = torch.tensor([[102]], dtype=torch.long)
    ids = torch.cat((cls, ids, sep), dim=1)

    mask = torch.zeros(1, 63, dtype=torch.long)
    token_type_ids = torch.zeros(1, 63, dtype=torch.long)
    traced = torch.jit.trace(model, ids, mask, token_type_ids)
    return traced


def optimizer(weights_path):
    metadata = joblib.load('metadata.bin')
    num_entity = len(metadata['enc_entity'].classes_)
    num_intent = len(metadata['enc_intent'].classes_)
    num_scenario = len(metadata['enc_scenario'].classes_)

    print("num_entity:", num_entity, "num_intent",
          num_intent, "num_scenario", num_scenario)

    model = NLUModel(num_entity, num_intent, num_scenario)
    model.load_state_dict(torch.load(weights_path,
                                     map_location=lambda storage, loc: storage))
    model.to(config.DEVICE)

    print('tracing model......')
    traced_model = trace(model)
    traced_model.save(config.TRACE_MODEL_PATH)
    print('Saved traced model to ', config.TRACE_MODEL_PATH)
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model Optimizer")
    parser.add_argument('--weights', type=str, default=config.MODEL_PATH, required=True,
                        help='Unoptimized Weights for Model.')
    args = parser.parse_args()

    optimizer(args.weights)
