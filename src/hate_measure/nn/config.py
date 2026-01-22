from transformers import AutoConfig, PretrainedConfig

class HateSpeechScorerConfig(PretrainedConfig):
    model_type = "hate-speech-scorer"
    auto_map = {
        "AutoConfig": "config.HateSpeechScorerConfig",
        "AutoModel": "scorer.HateSpeechScorer",
        "AutoModelForSequenceClassification": "scorer.HateSpeechScorer",
    }

    def __init__(
        self,
        encoder_model_name_or_path="AnswerDotAI/ModernBERT-base",
        n_dense=128,
        dropout_rate=0.4,
        num_labels=1,
        encoder_config=None,
        **kwargs
    ):
        super().__init__(num_labels=num_labels, **kwargs)
        self.problem_type = "regression"
        self.encoder_model_name_or_path = encoder_model_name_or_path
        self.n_dense = n_dense
        self.dropout_rate = dropout_rate
        self.id2label = {0: "score"}
        self.label2id = {"score": 0}
        # Use provided encoder_config (from saved config.json) or fetch fresh
        if encoder_config is not None:
            self.encoder_config = AutoConfig.from_pretrained(encoder_config["_name_or_path"])
        else:
            self.encoder_config = AutoConfig.from_pretrained(encoder_model_name_or_path)
