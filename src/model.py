from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Dict
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedModel, AutoModel, AutoConfig
from transformers.file_utils import ModelOutput


def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))


def masked_log_softmax(vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1) -> torch.Tensor:
    """
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a log_softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + tiny_value_of_dtype(vector.dtype)).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


def contrastive_loss(
    scores: torch.FloatTensor,
    positions: Union[List[int], Tuple[List[int], List[int]]],
    mask: torch.BoolTensor,
    prob_mask: torch.BoolTensor = None,
) -> torch.FloatTensor:
    batch_size, seq_length = scores.size(0), scores.size(1)
    if len(scores.shape) == 3:
        scores = scores.view(batch_size, -1)
        mask = mask.view(batch_size, -1)
        log_probs = masked_log_softmax(scores, mask)
        log_probs = log_probs.view(batch_size, seq_length, seq_length)
        start_positions, end_positions = positions
        batch_indices = list(range(batch_size))
        log_probs = log_probs[batch_indices, start_positions, end_positions]
    else:
        log_probs = masked_log_softmax(scores, mask)
        batch_indices = list(range(batch_size))
        log_probs = log_probs[batch_indices, positions]
    if prob_mask is not None:
        log_probs = log_probs * prob_mask
    return - log_probs.mean()


@dataclass
class BinderModelOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    start_scores: torch.FloatTensor = None
    end_scores: torch.FloatTensor = None
    span_scores: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Binder(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        hf_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            cache_dir=config.cache_dir,
            revision=config.revision,
            use_auth_token=config.use_auth_token,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )
        self.hf_config = hf_config
        self.config.pruned_heads = hf_config.pruned_heads
        self.dropout = torch.nn.Dropout(hf_config.hidden_dropout_prob)
        self.type_start_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.type_end_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.type_span_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.start_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        self.end_linear = torch.nn.Linear(hf_config.hidden_size, config.linear_size)
        if config.use_span_width_embedding:
            self.span_linear = torch.nn.Linear(hf_config.hidden_size * 2 + config.linear_size, config.linear_size)
            self.width_embeddings = torch.nn.Embedding(config.max_span_width, config.linear_size, padding_idx=0)
        else:
            self.span_linear = torch.nn.Linear(hf_config.hidden_size * 2, config.linear_size)
            self.width_embeddings = None
        self.start_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.end_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))
        self.span_logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / config.init_temperature))

        self.start_loss_weight = config.start_loss_weight
        self.end_loss_weight = config.end_loss_weight
        self.span_loss_weight = config.span_loss_weight
        self.threshold_loss_weight = config.threshold_loss_weight
        self.ner_loss_weight = config.ner_loss_weight

        # Initialize weights and apply final processing
        self.post_init()

        self.text_encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
            add_pooling_layer=False
        )
        self.type_encoder = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            config=hf_config,
            add_pooling_layer=False
        )

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.hf_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self):
        self.text_encoder.gradient_checkpointing_enable()
        self.type_encoder.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        type_input_ids: torch.LongTensor = None,
        type_attention_mask: torch.Tensor = None,
        type_token_type_ids: torch.Tensor = None,
        ner: Optional[Dict] = None,
        return_dict: bool = None,
    ):
        return_dict = return_dict if return_dict is not None else self.hf_config.use_return_dict

        outputs = self.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        # batch_size x seq_length x hidden_size
        sequence_output = outputs[0]

        type_outputs = self.type_encoder(
            type_input_ids.squeeze(0),
            attention_mask=type_attention_mask.squeeze(0),
            token_type_ids=type_token_type_ids.squeeze(0) if type_token_type_ids is not None else None,
            return_dict=return_dict,
        )
        # num_types x hidden_size
        type_output = type_outputs[0][:, 0]

        batch_size, seq_length, _ = sequence_output.size()
        num_types, _ = type_output.size()

        # num_types x hidden_size
        type_start_output = F.normalize(self.dropout(self.type_start_linear(type_output)), dim=-1)
        type_end_output = F.normalize(self.dropout(self.type_end_linear(type_output)), dim=-1)
        # batch_size x seq_length x hidden_size
        sequence_start_output = F.normalize(self.dropout(self.start_linear(sequence_output)), dim=-1)
        sequence_end_output = F.normalize(self.dropout(self.end_linear(sequence_output)), dim=-1)

        # batch_size x num_types x seq_length
        start_scores = self.start_logit_scale.exp() * type_start_output.unsqueeze(0) @ sequence_start_output.transpose(1, 2)
        end_scores = self.end_logit_scale.exp() * type_end_output.unsqueeze(0) @ sequence_end_output.transpose(1, 2)

        # batch_size x seq_length x seq_length x hidden_size*2
        span_output = torch.cat(
            [
                sequence_output.unsqueeze(2).expand(-1, -1, seq_length, -1),
                sequence_output.unsqueeze(1).expand(-1, seq_length, -1, -1),
            ],
            dim=3
        )

        # span_width_embeddings
        if self.width_embeddings is not None:
            range_vector = torch.cuda.LongTensor(seq_length, device=sequence_output.device).fill_(1).cumsum(0) - 1
            span_width = range_vector.unsqueeze(0) - range_vector.unsqueeze(1) + 1
            # seq_length x seq_length x hidden_size
            span_width_embeddings = self.width_embeddings(span_width * (span_width > 0))
            span_output = torch.cat([
                span_output, span_width_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)], dim=3)

        # batch_size x seq_length x seq_length x hidden_size
        span_linear_output = F.normalize(
            self.dropout(self.span_linear(span_output)).view(batch_size, seq_length * seq_length, -1), dim=-1
        )
        # num_types x hidden_size
        type_linear_output = F.normalize(self.dropout(self.type_span_linear(type_output)), dim=-1)

        span_scores = self.span_logit_scale.exp() * type_linear_output.unsqueeze(0) @ span_linear_output.transpose(1, 2)
        span_scores = span_scores.view(batch_size, num_types, seq_length, seq_length)

        total_loss = None
        if ner is not None:
            flat_start_scores = start_scores.view(batch_size * num_types, seq_length)
            flat_end_scores = end_scores.view(batch_size * num_types, seq_length)
            flat_span_scores = span_scores.view(batch_size * num_types, seq_length, seq_length)
            start_negative_mask = ner["start_negative_mask"].view(batch_size * num_types, seq_length)
            end_negative_mask = ner["end_negative_mask"].view(batch_size * num_types, seq_length)
            span_negative_mask = ner["span_negative_mask"].view(batch_size * num_types, seq_length, seq_length)

            start_threshold_loss = contrastive_loss(flat_start_scores, 0, start_negative_mask)
            end_threshold_loss = contrastive_loss(flat_end_scores, 0, end_negative_mask)
            span_threshold_loss = contrastive_loss(flat_span_scores, (0, 0), span_negative_mask)

            threshold_loss = (
                self.start_loss_weight * start_threshold_loss +
                self.end_loss_weight * end_threshold_loss +
                self.span_loss_weight * span_threshold_loss
            )

            ner_indices = ner["example_indices"]
            ner_starts, ner_ends = ner["example_starts"], ner["example_ends"]
            ner_start_masks, ner_end_masks = ner["example_start_masks"], ner["example_end_masks"]
            ner_span_masks = ner["example_span_masks"]

            start_loss = contrastive_loss(start_scores[ner_indices], ner_starts, ner_start_masks)
            end_loss = contrastive_loss(end_scores[ner_indices], ner_ends, ner_end_masks)
            span_loss = contrastive_loss(span_scores[ner_indices], (ner_starts, ner_ends), ner_span_masks)

            total_loss = (
                self.start_loss_weight * start_loss +
                self.end_loss_weight * end_loss +
                self.span_loss_weight * span_loss
            )

            total_loss = self.ner_loss_weight * total_loss + self.threshold_loss_weight * threshold_loss

        if not return_dict:
            output = (start_scores, end_scores, span_scores) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BinderModelOutput(
            loss=total_loss,
            start_scores=start_scores,
            end_scores=end_scores,
            span_scores=span_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
