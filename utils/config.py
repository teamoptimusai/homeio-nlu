import transformers

MAX_LEN = 61+2

BASE_MODEL = 'bert-base-uncased'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BASE_MODEL,
    do_lower_case=True
)
