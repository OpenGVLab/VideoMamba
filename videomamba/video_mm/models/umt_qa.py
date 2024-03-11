import logging

import torch
import torch.nn.functional as F
from einops import rearrange

from models.utils import tile

from .backbones.vit import build_vit
from .backbones.bert.builder import build_bert_decoder
from .umt import UMT

logger = logging.getLogger(__name__)


class UMT_QA(UMT):
    """docstring for UMT_QA"""

    def __init__(self, config, tokenizer, is_pretrain=False):
        super(UMT_QA, self).__init__(config, tokenizer, is_pretrain)

        # delete extra/unnecessary modules inherited from SingularityRetrievalBase
        extra_attributes = ["vision_proj", "text_proj", "temp", "itm_head"]
        for attr in extra_attributes:
            delattr(self, attr)

        self.text_decoder = self.build_text_decoder()

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.model.vision_encoder.name
        logger.info(f"Build vision_encoder: {encoder_name}")
        if "vit" in encoder_name:
            vision_encoder = build_vit(self.config.model, add_pool_norm=False)
        else:
            raise ValueError(f"not implemented: {encoder_name}")
        return vision_encoder

    def build_text_decoder(self):
        encoder_name = self.config.model.text_encoder.name
        logger.info(f"Build text_decoder {encoder_name}")
        if "bert" in encoder_name:
            text_decoder = build_bert_decoder(
                self.config.model, self.config.gradient_checkpointing
            )
        else:
            raise NotImplementedError()
        return text_decoder

    def encode_vision(self, image):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].

        """
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4) # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        keep_temporal = self.config.model.vision_encoder.keep_temporal
        vision_embeds, _ = self.vision_encoder(
            image, None, None, use_image, keep_temporal,
        )
        return vision_embeds

    def forward(self, image, question, answer=None, k=None, weights=None, train=True):
        """
        Args:
        k: number of answers for each question
        weights: weight for each answer
        """
        image_embeds = self.encode_vision(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        if train:
            answer_targets = answer.input_ids.masked_fill(
                answer.input_ids == self.tokenizer.pad_token_id, -100
            )

            question_output = self.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            question_states = []
            question_atts = []
            for b, n in enumerate(k):
                question_states += [question_output.last_hidden_state[b]] * n
                question_atts += [question.attention_mask[b]] * n
            question_states = torch.stack(question_states, 0)
            question_atts = torch.stack(question_atts, 0)

            answer_output = self.text_decoder(
                answer.input_ids,
                attention_mask=answer.attention_mask,
                encoder_hidden_states=question_states,
                encoder_attention_mask=question_atts,
                labels=answer_targets,
                return_dict=True,
                reduction="none",
            )
            loss = weights * answer_output.loss
            loss = loss.sum() / image.size(0)

            return loss

        # inference
        else:
            question_output = self.text_encoder(
                question.input_ids,
                attention_mask=question.attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            topk_ids, topk_probs = self.rank_answer(
                question_output.last_hidden_state,
                question.attention_mask,
                answer.input_ids,
                answer.attention_mask,
                k,
            )  # (bsz, 128), (bsz, 128)
            return topk_ids, topk_probs

    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        """
        question_states: (bsz, Lq, d)
        answer_ids: answer input id after tokenization, (#answers, La)
        """
        num_ques = question_states.size(0)
        start_ids = answer_ids[0, 0].repeat(num_ques, 1)  # bos token

        start_output = self.text_decoder(
            start_ids,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            return_dict=True,
            reduction="none",
        )
        logits = start_output.logits[:, 0, :]  # first token's logit

        # topk_probs: top-k probability
        # topk_ids: [num_question, k]
        answer_first_token = answer_ids[:, 1]
        prob_first_token = F.softmax(logits, dim=1).index_select(
            dim=1, index=answer_first_token
        )
        topk_probs, topk_ids = prob_first_token.topk(k, dim=1)

        # answer input: [num_question*k, answer_len]
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids, dim=0)
        input_atts = torch.cat(input_atts, dim=0)

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)

        output = self.text_decoder(
            input_ids,
            attention_mask=input_atts,
            encoder_hidden_states=question_states,
            encoder_attention_mask=question_atts,
            labels=targets_ids,
            return_dict=True,
            reduction="none",
        )

        answer_loss = output.loss
        answer_loss = answer_loss.view(input_ids.size(0), -1)

        # topk_prob: first token probability
        topk_probs = topk_probs.view(-1, 1)
        log_probs = torch.cat([topk_probs.log(), -answer_loss], dim=1)

        # re-calculate log probabilities for the answer sequences using chain rule
        log_probs_sum = log_probs.sum(1)
        log_probs_sum = log_probs_sum.view(num_ques, k)

        topk_probs = F.softmax(log_probs_sum, dim=-1)
        # get top-k after re-ranking
        topk_probs, rerank_id = topk_probs.topk(k, dim=1)
        topk_ids = torch.gather(topk_ids, 1, rerank_id)

        return topk_ids, topk_probs
