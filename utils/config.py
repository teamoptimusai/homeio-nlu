import transformers
import torch

RAW_DATASET_PATH = 'data/nlu.csv'
ER_DATASET_PATH = 'data/er_dataset.csv'
IS_DATASET_PATH = 'data/is_dataset.csv'
MODEL_PATH = ''
TRACE_MODEL_PATH = ''
LOG_PATH = 'runs/train/test_run'
SAVE_MODEL = True

MAX_LEN = 61+2
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
EPOCHS = 50

BASE_MODEL = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL,
    do_lower_case=True
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
