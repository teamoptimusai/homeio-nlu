import torch.nn as nn
import transformers
import utils.config as config


class NLUModel(nn.Module):
    def __init__(self, num_entity, num_intent, num_scenarios):
        super(NLUModel, self).__init__()
        self.num_entity = num_entity
        self.num_intent = num_intent
        self.num_scenario = num_scenarios

        self.bert = transformers.BertModel.from_pretrained(config.BASE_MODEL)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

        self.entity_out = nn.Linear(768, self.num_entity)
        self.intent_out = nn.Linear(768, self.num_intent)
        self.scenario_out = nn.Linear(768, self.num_scenario)

    def forward(self, ids, mask, token_type_ids):
        out = self.bert(input_ids=ids, attention_mask=mask,
                        token_type_ids=token_type_ids)

        hs, cls_hs = out['last_hidden_state'], out['pooler_output']

        entity_hs = self.dropout1(hs)
        intent_hs = self.dropout2(cls_hs)
        scenario_hs = self.dropout3(cls_hs)

        entity_hs = self.entity_out(entity_hs)
        intent_hs = self.intent_out(intent_hs)
        scenario_hs = self.scenario_out(scenario_hs)

        return entity_hs, intent_hs, scenario_hs
