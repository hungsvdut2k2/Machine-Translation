from transformers import BertTokenizerFast, GPT2TokenizerFast, AutoTokenizer


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def load_tokenizer(model_name_or_path):
    if "bert" in model_name_or_path.split("-"):
        return BertTokenizerFast.from_pretrained(model_name_or_path)
    elif "gpt2" in model_name_or_path.split("-"):
        return GPT2TokenizerFast.from_pretrained(model_name_or_path)
    else:
        return AutoTokenizer.from_pretrained(model_name_or_path)
