# coding: utf-8
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from itertools import groupby
from signjoey.initialization import initialize_model
from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import (
    Encoder,
    RecurrentEncoder,
    TransformerEncoder,
    DeformableTransformerEncoder,
    QueryTransformerEncoder,
)
from signjoey.decoders import (
    Decoder,
    RecurrentDecoder,
    TransformerDecoder,
    CrossTransformerDecoder,
)
from signjoey.search import beam_search, greedy
from signjoey.vocabulary import (
    TextVocabulary,
    GlossVocabulary,
    PAD_TOKEN,
    EOS_TOKEN,
    BOS_TOKEN,
)
from signjoey.batch import Batch
from signjoey.helpers import freeze_params, get_mask_from_sequence_lengths
from torch import Tensor
from typing import Union
from torchtext.vocab import Vectors
from pathlib import Path
import json
import pickle



class SignModel(nn.Module):
    """
    Base Model class
    """

    def __init__(
        self,
        encoder: Encoder,
        query_encoder: nn.Module,
        query_embedding: nn.Embedding,
        gloss_output_layer: nn.Module,
        decoder: Decoder,
        sgn_embed: SpatialEmbeddings,
        txt_embed: Embeddings,
        gls_vocab: GlossVocabulary,
        txt_vocab: TextVocabulary,
        name_to_video_id,
        video_cos_sim,
        do_recognition: bool = True,
        do_translation: bool = True,
        gloss_rate: int = 10,
        sim_loss_weight=0.0,
        sentence_embedding_mod="mean",
    ):
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param sgn_embed: spatial feature frame embeddings
        :param txt_embed: spoken language word embedding
        :param gls_vocab: gls vocabulary
        :param txt_vocab: spoken language vocabulary
        :param do_recognition: flag to build the model with recognition output.
        :param do_translation: flag to build the model with translation decoder.
        """
        super().__init__()
        self.name_to_video_id = name_to_video_id
        self.video_cos_sim = video_cos_sim
        self.encoder = encoder
        self.query_encoder = query_encoder
        self.decoder = decoder

        self.query_embedding = query_embedding
        self.sgn_embed = sgn_embed
        self.txt_embed = txt_embed

        self.gls_vocab = gls_vocab
        self.txt_vocab = txt_vocab

        self.txt_bos_index = self.txt_vocab.stoi[BOS_TOKEN]
        self.txt_pad_index = self.txt_vocab.stoi[PAD_TOKEN]
        self.txt_eos_index = self.txt_vocab.stoi[EOS_TOKEN]

        self.gloss_output_layer = gloss_output_layer
        self.do_recognition = do_recognition
        self.do_translation = do_translation
        self.gloss_rate = gloss_rate
        self.sim_loss_weight = sim_loss_weight
        self.sentence_embedding_mod = sentence_embedding_mod

    # pylint: disable=arguments-differ
    def forward(
        self,
        sgn: Tensor,
        sgn_mask: Tensor,
        sgn_lengths: Tensor,
        txt_input: Tensor,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param sgn: source input
        :param sgn_mask: source mask
        :param sgn_lengths: length of source inputs
        :param txt_input: target input
        :param txt_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self.encode(
            sgn=sgn, sgn_mask=sgn_mask, sgn_length=sgn_lengths, encoder=self.encoder
        )
        query_output = None
        query_mask = None

        if self.query_embedding is not None:
            if self.gloss_rate < 0:
                query_lens = sgn_lengths
            else:
                query_lens = sgn_lengths // self.gloss_rate

            query_lens = torch.where(
                query_lens < 1, query_lens.new_ones(query_lens.shape), query_lens
            )
            query_mask = get_mask_from_sequence_lengths(
                sequence_lengths=query_lens, device=sgn.device
            )
            query_index = (
                torch.arange(0, query_mask.shape[1], device=sgn.device)
                .unsqueeze(0)
                .expand(query_mask.shape)
            )
            query_vec = self.query_embedding(query_index)
            query_output = self.query_encoder(
                trg_embed=query_vec, encoder_output=encoder_output, src_mask=sgn_mask
            )

        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
        else:
            gloss_probabilities = None

        if self.do_translation:
            unroll_steps = txt_input.size(1)
            decoder_outputs = self.decode(
                encoder_output=encoder_output,
                encoder_output2=query_output,
                encoder_hidden=encoder_hidden,
                sgn_mask=sgn_mask,
                sgn_mask2=query_mask.unsqueeze(1) if query_mask is not None else None,
                txt_input=txt_input,
                unroll_steps=unroll_steps,
                txt_mask=txt_mask,
            )
        else:
            decoder_outputs = None

        return decoder_outputs, gloss_probabilities, encoder_output

    def encode(
        self, sgn: Tensor, sgn_mask: Tensor, sgn_length: Tensor, encoder
    ) -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param sgn:
        :param sgn_mask:
        :param sgn_length:
        :return: encoder outputs (output, hidden_concat)
        """
        return encoder(
            embed_src=self.sgn_embed(x=sgn, mask=sgn_mask),
            src_length=sgn_length,
            mask=sgn_mask,
        )

    def decode(
        self,
        encoder_output: Tensor,
        encoder_output2: Tensor,
        encoder_hidden: Tensor,
        sgn_mask: Tensor,
        sgn_mask2: Tensor,
        txt_input: Tensor,
        unroll_steps: int,
        decoder_hidden: Tensor = None,
        txt_mask: Tensor = None,
    ) -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param sgn_mask: sign sequence mask, 1 at valid tokens
        :param txt_input: spoken language sentence inputs
        :param unroll_steps: number of steps to unroll the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param txt_mask: mask for spoken language words
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(
            encoder_output=encoder_output,
            encoder_output2=encoder_output2,
            encoder_hidden=encoder_hidden,
            src_mask=sgn_mask,
            src_mask2=sgn_mask2,
            trg_embed=self.txt_embed(x=txt_input, mask=txt_mask),
            trg_mask=txt_mask,
            unroll_steps=unroll_steps,
            hidden=decoder_hidden,
        )

    def get_loss_for_batch(
        self,
        batch: Batch,
        recognition_loss_function: nn.Module,
        translation_loss_function: nn.Module,
        recognition_loss_weight: float,
        translation_loss_weight: float,
    ) -> (Tensor, Tensor):
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param recognition_loss_function: Sign Language Recognition Loss Function (CTC)
        :param translation_loss_function: Sign Language Translation Loss Function (XEntropy)
        :param recognition_loss_weight: Weight for recognition loss
        :param translation_loss_weight: Weight for translation loss
        :return: recognition_loss: sum of losses over sequences in the batch
        :return: translation_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable

        # Do a forward pass
        decoder_outputs, gloss_probabilities, encoder_output = self.forward(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_lengths=batch.sgn_lengths,
            txt_input=batch.txt_input,
            txt_mask=batch.txt_mask,
        )

        # Compute sim loss
        if self.sim_loss_weight > 0:
            batch_size, _, seq_len = batch.sgn_mask.shape
            sgn_mask = ~batch.sgn_mask.reshape(batch_size, seq_len, 1)
            if self.sentence_embedding_mod == "mean":
                # fill encoder_output with zeros
                encoder_output = encoder_output.masked_fill(sgn_mask, 0)
                sum_encoder_output = torch.sum(encoder_output, dim=1)
                sentence_emb = sum_encoder_output / batch.sgn_lengths.unsqueeze(1)
            elif self.sentence_embedding_mod == "max":
                encoder_output = encoder_output.masked_fill(sgn_mask, -float("inf"))
                sentence_emb = torch.max(encoder_output, dim=1)[0]
            sentence_emb = F.normalize(sentence_emb, dim=1)
            cos_sims = torch.mm(sentence_emb, sentence_emb.t())
            sim_labels = sentence_emb.new_zeros((batch_size, batch_size))
            sim_mask = torch.triu(
                sentence_emb.new_ones((batch_size, batch_size), dtype=bool), 1
            )
            for i in range(batch_size - 1):
                for j in range(i + 1, batch_size):
                    sim_labels[i, j] = self.video_cos_sim[
                        self.name_to_video_id[batch.sequence[i]],
                        self.name_to_video_id[batch.sequence[j]],
                    ]
            cos_sims = cos_sims.masked_select(sim_mask)
            sim_labels = sim_labels.masked_select(sim_mask)
            sim_loss = F.mse_loss(cos_sims, sim_labels)
            sim_loss = sim_loss * self.sim_loss_weight

        else:
            sim_loss = 0

        if self.do_recognition:
            assert gloss_probabilities is not None
            # Calculate Recognition Loss
            recognition_loss = (
                recognition_loss_function(
                    gloss_probabilities,
                    batch.gls,
                    batch.sgn_lengths.long(),
                    batch.gls_lengths.long(),
                )
                * recognition_loss_weight
            )
        else:
            recognition_loss = None

        if self.do_translation:
            assert decoder_outputs is not None
            word_outputs, _, _, _ = decoder_outputs
            # Calculate Translation Loss
            txt_log_probs = F.log_softmax(word_outputs, dim=-1)
            translation_loss = (
                translation_loss_function(txt_log_probs, batch.txt)
                * translation_loss_weight
            )
        else:
            translation_loss = None

        return recognition_loss, translation_loss, sim_loss

    def run_batch(
        self,
        batch: Batch,
        recognition_beam_size: int = 1,
        translation_beam_size: int = 1,
        translation_beam_alpha: float = -1,
        translation_max_output_length: int = 100,
    ) -> (np.array, np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param recognition_beam_size: size of the beam for CTC beam search
            if 1 use greedy
        :param translation_beam_size: size of the beam for translation beam search
            if 1 use greedy
        :param translation_beam_alpha: alpha value for beam search
        :param translation_max_output_length: maximum length of translation hypotheses
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        encoder_output, encoder_hidden = self.encode(
            sgn=batch.sgn,
            sgn_mask=batch.sgn_mask,
            sgn_length=batch.sgn_lengths,
            encoder=self.encoder,
        )
        query_output = None
        query_mask = None
        if self.query_embedding is not None:
            if self.gloss_rate < 0:
                query_lens = batch.sgn_lengths
            else:
                query_lens = batch.sgn_lengths // self.gloss_rate
            query_lens = torch.where(
                query_lens < 1, query_lens.new_ones(query_lens.shape), query_lens
            )
            query_mask = get_mask_from_sequence_lengths(
                sequence_lengths=query_lens, device=batch.sgn.device
            )
            query_index = (
                torch.arange(0, query_mask.shape[1], device=batch.sgn.device)
                .unsqueeze(0)
                .expand(query_mask.shape)
            )
            query_vec = self.query_embedding(query_index)
            query_output = self.query_encoder(
                trg_embed=query_vec,
                encoder_output=encoder_output,
                src_mask=batch.sgn_mask,
            )
        if self.do_recognition:
            # Gloss Recognition Part
            # N x T x C
            gloss_scores = self.gloss_output_layer(encoder_output)
            # N x T x C
            gloss_probabilities = gloss_scores.log_softmax(2)
            # Turn it into T x N x C
            gloss_probabilities = gloss_probabilities.permute(1, 0, 2)
            gloss_probabilities = gloss_probabilities.cpu().detach().numpy()
            tf_gloss_probabilities = np.concatenate(
                (gloss_probabilities[:, :, 1:], gloss_probabilities[:, :, 0, None]),
                axis=-1,
            )

            assert recognition_beam_size > 0
            ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
                inputs=tf_gloss_probabilities,
                sequence_length=batch.sgn_lengths.cpu().detach().numpy(),
                beam_width=recognition_beam_size,
                top_paths=1,
            )
            ctc_decode = ctc_decode[0]
            # Create a decoded gloss list for each sample
            tmp_gloss_sequences = [[] for i in range(gloss_scores.shape[0])]
            for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
                tmp_gloss_sequences[dense_idx[0]].append(
                    ctc_decode.values[value_idx].numpy() + 1
                )
            decoded_gloss_sequences = []
            for seq_idx in range(0, len(tmp_gloss_sequences)):
                decoded_gloss_sequences.append(
                    [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
                )
        else:
            decoded_gloss_sequences = None

        if self.do_translation:
            # greedy decoding
            if translation_beam_size < 2:
                stacked_txt_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    encoder_output2=query_output,
                    src_mask=batch.sgn_mask,
                    src_mask2=query_mask.unsqueeze(1)
                    if query_mask is not None
                    else None,
                    embed=self.txt_embed,
                    bos_index=self.txt_bos_index,
                    eos_index=self.txt_eos_index,
                    decoder=self.decoder,
                    max_output_length=translation_max_output_length,
                )
                # batch, time, max_sgn_length
            else:  # beam size
                stacked_txt_output, stacked_attention_scores = beam_search(
                    size=translation_beam_size,
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    encoder_output2=query_output,
                    src_mask=batch.sgn_mask,
                    src_mask2=query_mask.unsqueeze(1)
                    if query_mask is not None
                    else None,
                    embed=self.txt_embed,
                    max_output_length=translation_max_output_length,
                    alpha=translation_beam_alpha,
                    eos_index=self.txt_eos_index,
                    pad_index=self.txt_pad_index,
                    bos_index=self.txt_bos_index,
                    decoder=self.decoder,
                )
        else:
            stacked_txt_output = stacked_attention_scores = None

        return decoded_gloss_sequences, stacked_txt_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return (
            "%s(\n"
            "\tencoder=%s,\n"
            "\tdecoder=%s,\n"
            "\tsgn_embed=%s,\n"
            "\ttxt_embed=%s)"
            % (
                self.__class__.__name__,
                self.encoder,
                self.decoder,
                self.sgn_embed,
                self.txt_embed,
            )
        )


def build_model(
    cfg: dict,
    sgn_dim: int,
    gls_vocab: GlossVocabulary,
    txt_vocab: TextVocabulary,
    do_recognition: bool = True,
    do_translation: bool = True,
) -> SignModel:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param sgn_dim: feature dimension of the sign frame representation, i.e. 2560 for EfficientNet-7.
    :param gls_vocab: sign gloss vocabulary
    :param txt_vocab: spoken language word vocabulary
    :return: built and initialized model
    :param do_recognition: flag to build the model with recognition output.
    :param do_translation: flag to build the model with translation decoder.
    """

    txt_padding_idx = txt_vocab.stoi[PAD_TOKEN]

    sgn_embed: SpatialEmbeddings = SpatialEmbeddings(
        **cfg["encoder"]["embeddings"],
        num_heads=cfg["encoder"]["num_heads"],
        input_size=sgn_dim,
    )

    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.0)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "deformable_transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = DeformableTransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )
    elif cfg["encoder"].get("type", "recurrent") == "transformer":
        assert (
            cfg["encoder"]["embeddings"]["embedding_dim"]
            == cfg["encoder"]["hidden_size"]
        ), "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )
    else:
        encoder = RecurrentEncoder(
            **cfg["encoder"],
            emb_size=sgn_embed.embedding_dim,
            emb_dropout=enc_emb_dropout,
        )

    if cfg.get("query_encoder", ""):
        query_encoder = QueryTransformerEncoder(
            **cfg["query_encoder"],
            emb_dropout=enc_emb_dropout,
        )
        query_embedding = nn.Embedding(
            cfg.get("max_query_embedding", 100), sgn_embed.embedding_dim
        )
    else:
        query_encoder = None
        query_embedding = None

    if do_recognition:
        gloss_output_layer = nn.Linear(encoder.output_size, len(gls_vocab))
        if cfg["encoder"].get("freeze", False):
            freeze_params(gloss_output_layer)
    else:
        gloss_output_layer = None

    # build decoder and word embeddings
    if do_translation:
        txt_embed: Union[Embeddings, None] = Embeddings(
            **cfg["decoder"]["embeddings"],
            num_heads=cfg["decoder"]["num_heads"],
            hidden_size=cfg["decoder"]["hidden_size"],
            vocab_size=len(txt_vocab),
            padding_idx=txt_padding_idx,
        )
        dec_dropout = cfg["decoder"].get("dropout", 0.0)
        dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
        if cfg["decoder"].get("type", "recurrent") == "transformer":
            decoder = TransformerDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
        elif cfg["decoder"].get("type", "recurrent") == "CrossTransformerDecoder":
            decoder = CrossTransformerDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
        else:
            decoder = RecurrentDecoder(
                **cfg["decoder"],
                encoder=encoder,
                vocab_size=len(txt_vocab),
                emb_size=txt_embed.embedding_dim,
                emb_dropout=dec_emb_dropout,
            )
    else:
        txt_embed = None
        decoder = None
    sim_name_to_video_id_json = cfg.get("sim_name_to_video_id_json", "")
    sim_video_cos_sim = cfg.get("sim_video_cos_sim", "")
    assert sim_name_to_video_id_json != "" and sim_video_cos_sim != "", "sim_name_to_video_id_json and sim_video_cos_sim must be set"
    
    name_to_video_id = json.load(
        open(sim_name_to_video_id_json)
    )
    video_cos_sim = pickle.load(
        open(sim_video_cos_sim, "rb")
    )
    
    model: SignModel = SignModel(
        encoder=encoder,
        query_encoder=query_encoder,
        query_embedding=query_embedding,
        gloss_output_layer=gloss_output_layer,
        decoder=decoder,
        sgn_embed=sgn_embed,
        txt_embed=txt_embed,
        gls_vocab=gls_vocab,
        txt_vocab=txt_vocab,
        do_recognition=do_recognition,
        do_translation=do_translation,
        gloss_rate=cfg.get("gloss_rate", 10),
        sim_loss_weight=cfg.get("sim_loss_weight", 0.0),
        sentence_embedding_mod=cfg.get("sentence_embedding_mod", "mean"),
        name_to_video_id=name_to_video_id,
        video_cos_sim=video_cos_sim
    )

    if do_translation:
        # tie softmax layer with txt embeddings
        if cfg.get("tied_softmax", False):
            # noinspection PyUnresolvedReferences
            if txt_embed.lut.weight.shape == model.decoder.output_layer.weight.shape:
                # (also) share txt embeddings and softmax layer:
                # noinspection PyUnresolvedReferences
                model.decoder.output_layer.weight = txt_embed.lut.weight
            else:
                raise ValueError(
                    "For tied_softmax, the decoder embedding_dim and decoder "
                    "hidden_size must be the same."
                    "The decoder must be a Transformer."
                )

    # custom initialization of model parameters
    initialize_model(model, cfg, txt_padding_idx)

    if do_translation:
        vectors_path = cfg.get("word_vectors", "")
        if vectors_path:
            vectors_path = Path(vectors_path).expanduser()
            pretrain_vector = Vectors(
                name=vectors_path.name,
                cache=str(vectors_path.parent),
            )
            for i, token in enumerate(txt_vocab.itos):
                if token.strip() in pretrain_vector.stoi:
                    model.txt_embed.lut.weight.data[i][
                        : pretrain_vector.dim
                    ] = pretrain_vector[token.strip()]

    return model
