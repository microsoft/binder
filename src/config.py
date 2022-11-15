from transformers import PretrainedConfig

class BinderConfig(PretrainedConfig):

    def __init__(
        self,
        pretrained_model_name_or_path=None,
        cache_dir=None,
        revision="main",
        use_auth_token=False,
        hidden_dropout_prob=0.1,
        max_span_width=30,
        use_span_width_embedding=False,
        linear_size=128,
        init_temperature=0.07,
        start_loss_weight=0.2,
        end_loss_weight=0.2,
        span_loss_weight=0.6,
        threshold_loss_weight=0.5,
        ner_loss_weight=0.5,
    ):
        self.pretrained_model_name_or_path=pretrained_model_name_or_path
        self.cache_dir=cache_dir
        self.revision=revision
        self.use_auth_token=use_auth_token
        self.hidden_dropout_prob=hidden_dropout_prob
        self.max_span_width = max_span_width
        self.use_span_width_embedding = use_span_width_embedding
        self.linear_size = linear_size
        self.init_temperature = init_temperature
        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.span_loss_weight = span_loss_weight
        self.threshold_loss_weight = threshold_loss_weight
        self.ner_loss_weight = ner_loss_weight

        self.max_span_width = max_span_width
        self.use_span_width_embedding = use_span_width_embedding
        self.linear_size = linear_size
        self.init_temperature = init_temperature
        self.start_loss_weight = start_loss_weight
        self.end_loss_weight = end_loss_weight
        self.span_loss_weight = span_loss_weight
        self.threshold_loss_weight = threshold_loss_weight
        self.ner_loss_weight = ner_loss_weight
