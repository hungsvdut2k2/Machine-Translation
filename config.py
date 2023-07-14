class BaseConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class NMTConfig(BaseConfig):
    # Data
    src_lang = "vi"
    tgt_lang = "en"
    src_max_len = 75
    tgt_max_len = 75

    # Model
    src_model_name = "bert-base-multilingual-cased"
    tgt_model_name = "gpt2"

    # Training
    load_model_from_path = False
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    learning_rate = 3e-5
    train_batch_size = 32
    eval_batch_size = 32
    num_train_epochs = 25
    ckpt_dir = src_model_name + "_to_" + tgt_model_name
    use_eval_steps = False
    eval_steps = 2000

    # Inference
    max_length_decoder = 75
    min_length_decoder = 25
    beam_size = 5


cfg = NMTConfig()
