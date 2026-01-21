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
        base_model_name="AnswerDotAI/ModernBERT-base",
        n_dense=128,
        dropout_rate=0.4,
        num_labels=1,
        **kwargs
    ):
        super().__init__(num_labels=num_labels,**kwargs)
        self.problem_type = "regression"
        self.encoder_model_name_or_path = base_model_name
        self.n_dense = n_dense
        self.dropout_rate = dropout_rate
        self.id2label = {0: "score"}
        self.label2id = {"score": 0}
        self.encoder_config = AutoConfig.from_pretrained(base_model_name)
