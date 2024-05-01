from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizerFast
from transformers.models.bert.modeling_bert import BertEncoder, BertModel, \
    BertForMaskedLM, BertEmbeddings, BertPooler, BertOnlyMLMHead, \
    MaskedLMOutput, BaseModelOutputWithPastAndCrossAttentions, \
    BaseModelOutputWithPoolingAndCrossAttentions, logger


class AlterableBertEncoder(BertEncoder):
    """
    A BertEncoder whose representations you can alter.
    """

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=None, output_attentions=False,
                output_hidden_states=False, return_dict=True,
                alter_layer: Optional[int] = None,
                remove_features: Optional[torch.Tensor] = None,
                alter_mask: Optional[torch.LongTensor] = None,
                alpha: float = 4.):
        """
        This is an exact copy of BertEncoder.forward, except that you
        are now allowed to remove a feature from a particular layer's
        representations.

        :param alter_layer: The layer to be altered
        :param remove_features: The feature(s) to be removed
        :param alter_mask: A mask that excludes certain tokens from
            alteration
        :param alpha: The parameter from AlterRep
        """
        # New code: Validate arguments for new parameters
        if (alter_layer is None) != (remove_features is None):
            raise ValueError("alter_layer is {} but remove_features is {}"
                             "".format(alter_layer, remove_features))
        # End new code

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and \
                                     self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            # New code: alter the input to the layer
            if i == alter_layer:
                projections = (1 + alpha) * (hidden_states @ remove_features.T)
                offsets = \
                    (projections.unsqueeze(-1) * remove_features).sum(dim=2)
                if alter_mask is not None:
                    offsets = offsets * alter_mask.unsqueeze(-1)
                hidden_states = hidden_states - offsets
            # End new code

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] \
                if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient "
                        "checkpointing. Setting `use_cache=False`...")
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value,
                                      output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module), hidden_states,
                    attention_mask, layer_head_mask, encoder_hidden_states,
                    encoder_attention_mask)
            else:
                layer_outputs = layer_module(
                    hidden_states, attention_mask, layer_head_mask,
                    encoder_hidden_states, encoder_attention_mask,
                    past_key_value, output_attentions)

            hidden_states = layer_outputs[0]

            # New code: alter layer 12 hidden states
            if i == len(self.layer) - 1 and alter_layer == len(self.layer):
                projections = (1 + alpha) * (hidden_states @ remove_features.T)
                offsets = \
                    (projections.unsqueeze(-1) * remove_features).sum(dim=2)
                if alter_mask is not None:
                    offsets = offsets * alter_mask.unsqueeze(-1)
                hidden_states = hidden_states - offsets
            # End new code

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + \
                                           (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache,
                                     all_hidden_states, all_self_attentions,
                                     all_cross_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states, attentions=all_self_attentions,
            cross_attentions=all_cross_attentions)


class AlterableBertModel(BertModel):
    """
    An alterable BERT model
    """

    def __init__(self, config, add_pooling_layer=True):
        """
        This is an exact copy of BertModel.__init__, except that
        self.encoder is an AlterableBertEncoder.
        """
        super(BertModel, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = AlterableBertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None,
                alter_layer: Optional[int] = None,
                remove_features: Optional[torch.Tensor] = None,
                alter_mask: Optional[torch.LongTensor] = None,
                alpha: float = 4.):
        """
        An exact copy of BertModel.forward, except that new arguments
        are passed to self.encoder.
        """
        output_attentions = output_attentions \
            if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states \
            if output_hidden_states is not None else \
            self.config.output_hidden_states

        return_dict = return_dict if return_dict is not None else \
            self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else \
                self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and "
                             "inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else \
            inputs_embeds.device

        past_key_values_length = past_key_values[0][0].shape[2] \
            if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = \
                    self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = \
                    buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long,
                                             device=device)

        extended_attention_mask: torch.Tensor = \
            self.get_extended_attention_mask(attention_mask, input_shape,
                                             device)

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = \
                encoder_hidden_states.size()
            encoder_hidden_shape = \
                (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = \
                    torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = \
                self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        head_mask = \
            self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids,
            token_type_ids=token_type_ids, inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length)

        encoder_outputs = self.encoder(
            embedding_output, attention_mask=extended_attention_mask,
            head_mask=head_mask, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values, use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # New code: Pass new arguments to encoder
            alter_layer=alter_layer, remove_features=remove_features,
            alter_mask=alter_mask, alpha=alpha)
        # End new code

        sequence_output = encoder_outputs[0]
        pooled_output = \
            self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output, pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions)


class AlterableBertForMaskedLM(BertForMaskedLM):
    """
    An alterable BERT masked language model
    """

    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        if config.is_decoder:
            logger.warning("If you want to use `BertForMaskedLM` make sure "
                           "`config.is_decoder=False` for bi-directional "
                           "self-attention.")

        self.bert = AlterableBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                labels=None, output_attentions=None, output_hidden_states=None,
                return_dict=None, alter_layer: Optional[int] = None,
                remove_features: Optional[torch.Tensor] = None,
                alter_mask: Optional[torch.LongTensor] = None,
                alpha: float = 4.):
        """
        Exact copy of BertForMaskedLM.forward, except that new arguments
        are passed to self.bert.
        """
        return_dict = return_dict if return_dict is not None else \
            self.config.use_return_dict

        outputs = self.bert(
            input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids, position_ids=position_ids,
            head_mask=head_mask, inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # New code: Pass new arguments
            alter_layer=alter_layer, remove_features=remove_features,
            alter_mask=alter_mask, alpha=alpha)
        # End new code

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = \
                loss_fct(prediction_scores.view(-1, self.config.vocab_size),
                         labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) \
                if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss, logits=prediction_scores,
            hidden_states=outputs.hidden_states, attentions=outputs.attentions)


if __name__ == "__main__":
    model = AlterableBertForMaskedLM.from_pretrained("bert-base-uncased")
    feature = torch.randn(3, model.config.hidden_size)
    feature = F.normalize(feature, dim=-1)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    input_ = tokenizer("Hello world!", return_tensors="pt")
    print(model(**input_, alter_layer=2, remove_features=feature))
