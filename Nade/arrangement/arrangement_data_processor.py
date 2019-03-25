import random

import music21
import torch
from torch import nn

from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from DatasetManager.helpers import PAD_SYMBOL, REST_SYMBOL, END_SYMBOL, START_SYMBOL

from Transformer.data_processor import DataProcessor
from Transformer.helpers import cuda_variable
from DatasetManager.arrangement.arrangement_helper import score_to_pianoroll, quantize_velocity_pianoroll_frame


class ArrangementDataProcessor(DataProcessor):
    def __init__(self, dataset: ArrangementDataset,
                 embedding_dim,
                 reducer_input_dim,
                 local_position_embedding_dim,
                 flag_orchestra,
                 block_attention
                 ):
        """

        :param dataset:
        :param embedding_dim:
        :param reducer_input_dim: dim before applying linear layers of
        different size (one for each voice) to obtain the correct shapes before
        softmax
        :param local_position_embedding_dim:

        """
        # todo note: reducer_input_dim = d_model
        # local_position_dim = d_ticks
        super(ArrangementDataProcessor, self).__init__(dataset=dataset,
                                                       embedding_dim=embedding_dim)

        self.dataset = dataset
        self.flag_orchestra = flag_orchestra
        self.block_attention = block_attention

        # Useful parameters
        self.num_frames_orchestra = int(
            (dataset.sequence_size + 1) / 2)  # Only past and present is used for the orchestration
        self.num_instruments = dataset.number_instruments
        #
        self.num_frames_piano = dataset.sequence_size  # For the piano, future is also used
        self.num_pitch_piano = dataset.number_pitch_piano

        self.num_notes_per_instrument_orchestra = [len(v) for k, v in self.dataset.index2midi_pitch.items()]
        self.num_notes_per_instrument_piano = [len(v) for k, v in self.dataset.value2oneHot_perPianoToken.items()]

        # Generic names
        self.max_len_sequence_decoder = self.num_instruments * self.num_frames_orchestra
        self.max_len_sequence_encoder = self.num_pitch_piano * self.num_frames_piano
        if self.flag_orchestra:
            self.num_token_per_tick = self.num_notes_per_instrument_orchestra
            self.num_ticks_per_frame = self.num_instruments
            self.num_frames = self.num_frames_orchestra
            max_len_sequence = self.max_len_sequence_decoder
        else:
            self.num_token_per_tick = self.num_notes_per_instrument_piano
            self.num_ticks_per_frame = self.num_pitch_piano
            self.num_frames = self.num_frames_piano
            max_len_sequence = self.max_len_sequence_encoder

        self.local_position_embedding_dim = local_position_embedding_dim

        self.note_embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings, self.embedding_dim)
                for num_embeddings in self.num_token_per_tick
            ]
        )

        self.linear_output_notes = nn.ModuleList(
            [
                nn.Linear(reducer_input_dim, num_notes)
                for num_notes in self.num_token_per_tick
            ]
        )

        # labels and embeddings for local position
        self.instrument_labels = nn.Parameter(
            torch.Tensor([(i % self.num_ticks_per_frame) for i in range(max_len_sequence)]).long(),
            requires_grad=False)
        self.instrument_embedding = nn.Embedding(self.num_ticks_per_frame,
                                                 local_position_embedding_dim)
        return

    def preprocessing(self, *tensors):
        """
        Discards metadata
        :param tensors:
        :return:
        """
        piano, orchestra = tensors
        # Here extract information used by the network
        if self.flag_orchestra:
            # Use only past and present
            orchestra_past_present = orchestra[:, :self.num_frames_orchestra]
            return cuda_variable(orchestra_past_present.long())
        else:
            return cuda_variable(piano.long())

    def get_len_max_seq(self):
        return self.max_len_sequence_decoder if self.flag_orchestra else self.max_len_sequence_encoder

    def embed(self, x):
        """
        :param x: (batch_size, time, num_voices)
        :return: seq: (batch_size, time * num_voices, embedding_size)
        """
        shape_x = x.size()
        if len(shape_x) == 2:
            #  Deal with the case of single piano frame
            x_permute = x.permute(1, 0)
            x_embeds = [
                embedding(voice)
                for voice, embedding
                in zip(x_permute, self.note_embeddings)
            ]
            stacked_embedding = torch.stack(x_embeds)
            ret = stacked_embedding.permute(1, 0, 2)
        else:
            x_permute = x.permute(2, 0, 1)
            x_embeds = [
                embedding(voice)
                for voice, embedding
                in zip(x_permute, self.note_embeddings)
            ]
            stacked_embedding = torch.stack(x_embeds).permute(1, 2, 0, 3)
            ret = self.flatten(stacked_embedding)
        return ret

    def flatten(self, x):
        """
        :param x:(batch, num_frames, num_instruments, ...)
        :return: (batch, num_frames * num_instruments, ...) with num_instruments varying faster
        """
        batch_size = x.size()[0]
        x = torch.reshape(x, (batch_size, self.num_frames * self.num_ticks_per_frame, -1))
        return x

    def wrap(self, flatten_x):
        """
        Inverse of flatten operation

        :param flatten_x: (batch_size, length, ...)
        :return:
        """
        batch_size = flatten_x.size(0)
        x = torch.reshape(flatten_x, (batch_size, self.num_frames, self.num_ticks_per_frame, -1))
        return x

    def mask(self, x, p=None):
        if p is None:
            # TODO instead of random activation proba, use scheduling (start training with p low, finish with p high) with
            #   p increasing progressively
            p = random.random() / 2 + 0.5

        batch_size, num_frames, num_instruments = x.size()
        mask = (torch.rand_like(x.float()) < p).long()  #  1 means masked
        pad_matrix = cuda_variable(self.dataset.precomputed_vectors_orchestra[PAD_SYMBOL].long())
        pad_matrix = pad_matrix.unsqueeze(0).repeat(num_frames, 1)
        pad_matrix = pad_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        #  Todo: PROBLEM equivalent to using 0 as a padding information
        masked_chorale = x.clone() * (1 - mask) + pad_matrix.clone() * mask
        return masked_chorale, mask

    # @staticmethod
    # def mean_crossentropy(preds, targets, ratio=None):
    #     """
    #     The mean cross_entropy is computed only one the last temporal frame of the orchestra
    #     :param mask:
    #     :param ratio:
    #     :param targets: (batch, voice, chorale_length)
    #     :param preds: (num_instru, batch, nu_frames, num_notes) one for each voice
    #     since num_notes are different
    #     :return:
    #     """
    #     cross_entropy = nn.CrossEntropyLoss(size_average=True)
    #     sum = 0
    #     # Model learns to predict only the last frame index
    #     # And put voice first for targets
    #     targets_t = targets[:, -1, :].permute(1, 0)
    #     for pred, target_t in zip(preds, targets_t):
    #         pred_t = pred[:, -1, :]
    #         ce = cross_entropy(pred_t, target_t)
    #         sum += ce
    #     return sum

    @staticmethod
    def mean_crossentropy(preds, targets, mask, shift):
        """
        The mean cross_entropy is computed only one the last temporal frame of the orchestra
        :param mask:
        :param ratio:
        :param targets: (batch, voice, chorale_length)
        :param preds: (num_instru, batch, nu_frames, num_notes) one for each voice
        since num_notes are different
        :return:
        """
        # Todo reintroduire le truc du ratio pour privilegier les dernieres frames
        #  (qui sont en pratiques celles que l'on va devoir generer)

        if mask is None:
            reduction = 'mean'
        else:
            reduction = 'none'

        cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
        loss = 0

        batch_size, length, num_instruments = targets.size()

        targets_permute = targets.permute(2, 0, 1)
        if mask is not None:
            mask_permute = mask.permute(2, 0, 1)

        for instrument_index, this_pred in enumerate(preds):
            this_target = targets_permute[instrument_index]
            if mask is not None:
                this_mask = mask_permute[instrument_index].float()

            if shift and (instrument_index == 0):
                # If shifted sequences, don't use first prediction
                this_target = this_target[:, 1:]
                this_pred = this_pred[:, 1:, :]
                if mask is not None:
                    this_mask = this_mask[:, 1:]
                flat_dim = batch_size * (length - 1)
            else:
                flat_dim = batch_size * length

            this_targ_flat = this_target.view(flat_dim)
            this_pred_flat = this_pred.view(flat_dim, -1)
            if mask is not None:
                this_mask_flat = this_mask.view(flat_dim)

            # Padding mask is one where input is masked, and these are the samples we want to use to backprop
            ce = cross_entropy(this_pred_flat, this_targ_flat)
            if mask is not None:
                # Normalise by the number of elements actually used
                norm = this_mask_flat.sum()
                ce = torch.dot(ce, this_mask_flat) / norm

            loss += ce

        return loss

    def pred_seq_to_preds(self, pred_seq):
        """
        :param pred_seq: predictions = (b, l, num_features)
        :return: preds: (num_instru, batch, num_frames, num_pitches)
        """
        batch_size, length, num_features = pred_seq.size()

        assert length % self.num_instruments == 0
        # split voices
        pred_seq = pred_seq.view(batch_size, length // self.num_instruments,
                                 self.num_instruments, num_features).permute(2, 0, 1, 3)
        preds = [
            pre_softmax(pred)
            for pred, pre_softmax in zip(pred_seq, self.linear_output_notes)
        ]
        return preds

    def local_position(self, batch_size, sequence_length):
        """

        :param batch_size:
        :param sequence_length:
        :return:
        """
        num_repeats = (sequence_length + 1) // self.num_ticks_per_frame
        positions = self.instrument_labels.unsqueeze(0).repeat(batch_size, num_repeats)[:, :sequence_length]
        embedded_positions = self.instrument_embedding(positions)
        return embedded_positions

    def get_relative_attention_module(self, embedding_dim,
                                      n_head,
                                      max_len_seq,
                                      use_masks,
                                      shift,
                                      enc_dec_attention):
        if self.flag_orchestra:
            if enc_dec_attention:
                seq_len_encoder = self.max_len_sequence_encoder
                seq_len_decoder = self.max_len_sequence_decoder
                assert (max_len_seq == seq_len_decoder)
                return NoPositionwiseAttention(seq_len_enc=seq_len_encoder,
                                               seq_len_dec=max_len_seq)
            else:
                if self.block_attention:
                    return BlockSelfAttentionModule(embedding_dim=embedding_dim,
                                                    n_head=n_head,
                                                    num_instruments=self.num_instruments,
                                                    num_frames_orchestra=self.num_frames_orchestra,
                                                    use_masks=use_masks,
                                                    shift=shift)
                else:
                    return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                                                       n_head=n_head,
                                                       seq_len=max_len_seq,
                                                       use_masks=use_masks)
        else:
            return RelativeSelfAttentionModule(embedding_dim=embedding_dim,
                                               n_head=n_head,
                                               seq_len=max_len_seq,
                                               use_masks=use_masks)

    def init_generation_filepath(self, batch_size, filepath, banned_instruments=[]):
        # Get pianorolls
        score_piano = music21.converter.parse(filepath)
        pianoroll_piano, onsets_piano, _ = score_to_pianoroll(score_piano,
                                                              self.dataset.subdivision,
                                                              self.dataset.simplify_instrumentation,
                                                              self.dataset.transpose_to_sounding_pitch)

        quantized_pianoroll_piano = quantize_velocity_pianoroll_frame(pianoroll_piano["Piano"],
                                                                      self.dataset.velocity_quantization)
        onsets_piano = onsets_piano["Piano"]

        #  New events piano
        def new_events(pianoroll_piano):
            num_frames = pianoroll_piano.shape[0]
            events_indices = []
            frames_to_keep = []
            current_frame = None
            for frame_index in range(num_frames):
                if pianoroll_piano[frame_index].sum() == 0:
                    continue
                if (current_frame is None) or not (current_frame == pianoroll_piano[frame_index]).all():
                    events_indices.append(frame_index)
                    current_frame = pianoroll_piano[frame_index]
                    frames_to_keep.append(torch.from_numpy(current_frame))
            return events_indices, frames_to_keep

        rhythm_piano, _ = new_events(quantized_pianoroll_piano)
        piano_tensor = []
        for frame_index in rhythm_piano:
            piano_t_encoded = self.dataset.pianoroll_to_piano_tensor(
                quantized_pianoroll_piano,
                onsets_piano,
                frame_index)
            piano_tensor.append(piano_t_encoded)

        # Prepend padding frames at the beginning and end of the piano score
        piano_tensor = [self.dataset.precomputed_vectors_piano[REST_SYMBOL]] * (self.num_frames_orchestra - 1) + \
                       piano_tensor + \
                       [self.dataset.precomputed_vectors_piano[REST_SYMBOL]] * (self.num_frames_orchestra - 1)
        piano_init = torch.stack(piano_tensor)

        # Orchestra
        num_frames = piano_init.shape[0]  #  Here batch size is time dimensions (each batch index is a piano event)
        orchestra_silences, orchestra_init = self.init_orchestra(num_frames, banned_instruments)

        # Repeat along batch dimension to generate several orchestation of the same piano score
        piano_init = piano_init.unsqueeze(0).repeat(batch_size, 1, 1)
        orchestra_init = orchestra_init.unsqueeze(0).repeat(batch_size, 1, 1)

        return cuda_variable(piano_init.long()), rhythm_piano, \
               cuda_variable(orchestra_init.long()), orchestra_silences

    def init_generation(self, number_samples_per_chord=8, banned_instruments=[]):
        # #  C major and inversions
        # C_major = music21.chord.Chord(["C4", "E4", "G4"])
        # C_major_1 = music21.chord.Chord(["E4", "G4", "C5"])
        # C_major_2 = music21.chord.Chord(["G3", "C4", "E4"])
        # C7 = music21.chord.Chord(["C4", "E4", "G4", "B-4"])
        # # Tranpose the previous chords
        # transpositions = [
        #     music21.interval.Interval(2),
        #     music21.interval.Interval(3),
        #     music21.interval.Interval(6),
        #     music21.interval.Interval(7),
        # ]
        # base_chords = [C_major, C_major_1, C_major_2, C7]
        # transposed_chords = []
        # for transposition in transpositions:
        #     this_transposed_chords = [chord.transpose(transposition) for chord in base_chords]
        #     transposed_chords.extend(this_transposed_chords)
        # all_chords = base_chords + transposed_chords
        #
        # piano_dim = self.dataset.piano_tessitura['highest_pitch'] - self.dataset.piano_tessitura['lowest_pitch'] + 1
        # piano_score = []
        # for chord in all_chords:
        #     this_chord_vector = torch.zeros(piano_dim)
        #     for note in chord._notes:
        #         index = note_to_midiPitch(note) - self.dataset.piano_tessitura['lowest_pitch']
        #         this_chord_vector[index] = 1
        #     piano_score += ([this_chord_vector, ] * number_samples_per_chord)
        # piano_score = torch.stack(piano_score)
        #
        # rhythm_piano = list(range(len(piano_score)))
        #
        # batch_size = piano_score.shape[0]
        #
        # orchestra_silences, orchestra_init = self.__init_orchestra(batch_size, banned_instruments)
        #
        # return cuda_variable(piano_score.long()), rhythm_piano, \
        #        cuda_variable(orchestra_init.long()), orchestra_silences
        return

    def init_orchestra(self, num_frames, banned_instruments):
        # Set orchestra constraints in the form of banned instruments
        orchestra_silences = []
        orchestra_init = torch.zeros(num_frames, len(self.num_notes_per_instrument_orchestra))
        for instrument_name, instrument_indices in self.dataset.instrument2index.items():
            for instrument_index in instrument_indices:
                if instrument_name in banned_instruments:
                    # -1 is a silence
                    orchestra_silences.append(1)
                    orchestra_init[:, instrument_index] = self.dataset.midi_pitch2index[instrument_index][REST_SYMBOL]
                else:
                    orchestra_silences.append(0)
                    #  Initialise with last
                    orchestra_init[:, instrument_index] = self.dataset.midi_pitch2index[instrument_index][PAD_SYMBOL]

        # Start and end symbol at the beginning and end
        context_length = self.num_frames_orchestra - 1
        orchestra_init[:context_length] = self.dataset.precomputed_vectors_orchestra[START_SYMBOL]
        orchestra_init[context_length:] = self.dataset.precomputed_vectors_orchestra[END_SYMBOL]
        return orchestra_silences, orchestra_init

# OLD ONE
# DOES NOT USE THE TRICK FOR MAGENTA
# TODO adapt signature
# AND IT USED ONLY ONE SHARED R AMONGST THE HEADS
# class BachRelativeAttentionModule(nn.Module):
#     def __init__(self, embedding_dim, max_len_seq):
#         # todo embedding dim = d_k
#         super(BachRelativeAttentionModule, self).__init__()
#         self.embedding_dim = embedding_dim
#         # share R between heads
#         self.w_rs_tick = nn.Embedding(max_len_seq // 2 + 1, self.embedding_dim,
#                                       padding_idx=0)
#         self.w_rs_mod = nn.Embedding(17 + 1, self.embedding_dim,
#                                      padding_idx=0)
#
#         # TODO don't use triangular r?!
#         # self.r_index_tick = torch.Tensor([[(i // 4 - j // 4) + 1 if i - j >= 0 else 0 for i in
#         #                                    range(
#         #                                        get_len_max_seq)]
#         #                                   for j in range(get_len_max_seq)]).long().to('cuda')
#         self.r_index_tick = torch.Tensor([[(i // 4 - j // 4) + max_len_seq // 4 + 1 for i in
#                                            range(max_len_seq)]
#                                           for j in range(max_len_seq)]).long().to('cuda')
#         # self.r_index_mod = torch.Tensor([[((i % 4) * 4 + (j % 4)) if i - j >= 0 else -1 for i in
#         #                                   range(
#         #                                       get_len_max_seq)]
#         #                                  for j in range(get_len_max_seq)]).long().to('cuda')
#         # not the best idea:
#         undirected_order_dict = {(0, 0): 0,
#                                  (0, 1): 1,
#                                  (1, 0): 1,
#                                  (0, 2): 2,
#                                  (2, 0): 2,
#                                  (0, 3): 3,
#                                  (3, 0): 3,
#                                  (1, 1): 4,
#                                  (1, 2): 5,
#                                  (2, 1): 5,
#                                  (1, 3): 6,
#                                  (3, 1): 6,
#                                  (2, 2): 7,
#                                  (2, 3): 8,
#                                  (3, 2): 8,
#                                  (3, 3): 9,
#                                  }
#         # self.r_index_mod = torch.Tensor([[undirected_order_dict[(i % 4), (j % 4)] + 1 if i - j >= 0
#         #                                   else 0 for i in
#         #                                   range(
#         #                                       get_len_max_seq)]
#         #                                  for j in range(get_len_max_seq)]).long().to('cuda')
#         self.r_index_mod = torch.Tensor([[undirected_order_dict[(i % 4), (j % 4)] + 1
#                                           for i in
#                                           range(
#                                               max_len_seq)]
#                                          for j in range(max_len_seq)]).long().to('cuda')
#
#     def forward(self, sequence_length, n_head, batch_size):
#         # r is shared amongst the heads
#         r_mod = self.w_rs_mod(self.r_index_mod[:sequence_length, :sequence_length]).view(
#             sequence_length, sequence_length, self.embedding_dim)
#         r_tick = self.w_rs_tick(self.r_index_tick[:sequence_length, :sequence_length]).view(
#             sequence_length, sequence_length, self.embedding_dim)
#
#         r_mod = r_mod.unsqueeze(0).unsqueeze(0).repeat(n_head, batch_size, 1, 1, 1, 1)
#         r_mod = r_mod.view(-1, sequence_length, sequence_length,
#                            self.embedding_dim)  # (n*b) x lq x lq x dk
#
#         r_tick = r_tick.unsqueeze(0).unsqueeze(0).repeat(n_head, batch_size, 1, 1, 1, 1)
#         r_tick = r_tick.view(-1, sequence_length, sequence_length,
#                              self.embedding_dim)  # (n*b) x lq x lq x dk
#
#         r = r_mod + r_tick
#         return r


# Class copied from LsdbRelativeAttentionModule
class RelativeSelfAttentionModule(nn.Module):
    def __init__(self, embedding_dim, n_head, seq_len, use_masks):
        # todo
        #  TIED PARAMETERS
        #  embedding dim = d_k
        super(RelativeSelfAttentionModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.use_masks = use_masks

        if self.use_masks:
            self.e = nn.Parameter(
                torch.randn((n_head * seq_len), self.embedding_dim))
        else:
            self.e1 = nn.Parameter(
                torch.randn((n_head * seq_len), self.embedding_dim))
            self.e2 = nn.Parameter(
                torch.randn((n_head * seq_len), self.embedding_dim))

    def forward(self, q, flipped_masks):
        """

        :param q: (batch_size * n_head, len_q, d)
        :return:
        """
        sz_b_times_n_head, len_q, d_q = q.size()
        assert sz_b_times_n_head % self.n_head == 0
        sz_b = sz_b_times_n_head // self.n_head

        if self.use_masks:
            # In this case,
            # rel_attn will be masked in RelativeScaledDotProductAttention
            # the size of e can be larger that the size of q
            # todo add shift argument?!
            e = self.e[:self.n_head * len_q, :]
            e = e.unsqueeze(0).repeat(sz_b, 1, 1)
            e = e.view(sz_b * self.n_head, len_q, d_q)

            rel_attn = torch.einsum('bld,bmd->blm',
                                    (q, e))
            batch_size, l, _ = rel_attn.size()
            # skewing trick
            if flipped_masks:
                # pad
                rel_attn = torch.cat(
                    [rel_attn,
                     cuda_variable(torch.ones(batch_size,
                                              l,
                                              1,
                                              ) * - 100),
                     ], dim=2
                )
                rel_attn = rel_attn.view(batch_size,
                                         l + 1,
                                         l,
                                         )

                rel_attn = rel_attn[:, :-1, :]
            else:
                # todo refine
                # pad
                rel_attn = torch.cat(
                    [cuda_variable(torch.ones(batch_size,
                                              l,
                                              1,
                                              ) * - 100),
                     rel_attn
                     ], dim=2
                )
                rel_attn = rel_attn.view(batch_size,
                                         l + 1,
                                         l,
                                         )

                rel_attn = rel_attn[:, 1:, :]

        # if no mask is used
        # the trick must be done twice
        else:
            e1 = self.e1.unsqueeze(0).repeat(sz_b, 1, 1)
            e1 = e1.view(sz_b * self.n_head, len_q, d_q)

            rel_attn_1 = torch.einsum('bld,bmd->blm',
                                      (q, e1))
            e2 = self.e2.unsqueeze(0).repeat(sz_b, 1, 1)
            e2 = e2.view(sz_b * self.n_head, len_q, d_q)

            rel_attn_2 = torch.einsum('bld,bmd->blm',
                                      (q, e2))

            batch_size, l, _ = rel_attn_1.size()
            # ====skewing trick
            # ----Down
            # pad
            rel_attn_1 = torch.cat(
                [rel_attn_1,
                 cuda_variable(torch.ones(batch_size,
                                          l,
                                          1,
                                          ) * - 100),
                 ], dim=2
            )
            rel_attn_1 = rel_attn_1.view(batch_size,
                                         l + 1,
                                         l,
                                         )

            rel_attn_1 = rel_attn_1[:, :-1, :]

            # ----Up

            # pad
            rel_attn_2 = torch.cat(
                [cuda_variable(torch.ones(batch_size,
                                          l,
                                          1,
                                          ) * - 100),
                 rel_attn_2
                 ], dim=2
            )
            rel_attn_2 = rel_attn_2.view(batch_size,
                                         l + 1,
                                         l,
                                         )

            rel_attn_2 = rel_attn_2[:, 1:, :]

            masks_down = torch.triu(torch.ones_like(rel_attn_1[0]).byte(),
                                    diagonal=0).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1).flip(
                1).flip(2)
            masks_up = torch.triu(torch.ones_like(rel_attn_2[0]).byte(),
                                  diagonal=1).unsqueeze(0).repeat(sz_b_times_n_head, 1, 1)

            rel_attn_1 = rel_attn_1.masked_fill(masks_down, 0)
            rel_attn_2 = rel_attn_2.masked_fill(masks_up, 0)
            rel_attn = rel_attn_1 + rel_attn_2
        return rel_attn


class NoPositionwiseAttention(nn.Module):
    #  todo
    #  Zero attention module here ? Perhaps try the block matrix stuff also here
    def __init__(self, seq_len_enc, seq_len_dec):
        super(NoPositionwiseAttention, self).__init__()
        self.seq_len_enc = seq_len_enc
        return

    def forward(self, q, flipped_masks):
        """

        :param q: (batch_size * n_head, len_q, d)
        :return:
        """
        sz_b_times_n_head, len_q, d_q = q.size()
        rel_attn = torch.zeros(sz_b_times_n_head, len_q, self.seq_len_enc)
        return cuda_variable(rel_attn)


class BlockSelfAttentionModule(nn.Module):
    def __init__(self, embedding_dim, n_head, num_instruments, num_frames_orchestra, use_masks, shift):
        super(BlockSelfAttentionModule, self).__init__()

        self.n_head = n_head
        self.num_frames_orchestra = num_frames_orchestra
        self.num_instruments = num_instruments
        self.length = num_frames_orchestra * num_instruments
        self.embedding_dim = embedding_dim
        self.shift = shift
        self.use_masks = use_masks

        # Parameters
        # TODO We might want to have the same embedding for same instruments (which is not the case here)
        #  Like flute0 -> flute1 is the same as flute0 <-> flute1
        self.r_instrument = nn.Parameter(torch.randn(num_instruments, num_instruments, embedding_dim, n_head))
        self.e_past = nn.Parameter(torch.randn(num_frames_orchestra, embedding_dim, n_head))
        if not use_masks:
            #  Also need negatives relative values in that case
            self.e_future = nn.Parameter(torch.randn(num_frames_orchestra, embedding_dim, n_head))
        return

    def preprocess(self):

        #  Masks for grouping e0 and e1
        if not self.use_masks:
            one = torch.ones(self.num_frames_orchestra, self.num_frames_orchestra)
            mask_past = cuda_variable(torch.tril(one, diagonal=0)
                                           .unsqueeze(2).unsqueeze(3)
                                           .repeat(1, 1, self.embedding_dim, self.n_head))
            mask_future = cuda_variable(torch.tril(one, diagonal=-1)
                                             .unsqueeze(2).unsqueeze(3)
                                             .repeat(1, 1, self.embedding_dim, self.n_head))
            # Note that with the way we compute S_rel, one value of e_future is "wasted"
        else:
            mask_past = None

        # TEST VALUES
        # r_instrument = torch.zeros_like(self.r_instrument)
        # e_past = torch.zeros_like(self.e_past)
        # e_future = torch.zeros_like(self.e_future)
        # for i in range(self.num_instruments):
        #     for j in range(self.num_instruments):
        #         r_instrument[i, j, :, :] = (i + 1) * 100 + (j + 1) * 10
        # for i in range(self.num_frames_orchestra):
        #     e_past[i, :, ] = i + 1
        #     e_future[i, :, ] = (i + 1) * 10

        r_instrument = self.r_instrument
        e_past = self.e_past
        if not self.use_masks:
            e_future = self.e_future

        r_instrument_repeat = r_instrument.repeat(self.num_frames_orchestra, self.num_frames_orchestra, 1, 1)
        if self.shift:
            # Remove first column and last row
            r_instrument_repeat = r_instrument_repeat[1:, :-1]

        def prepare_r_time(inp, mask):
            # Triangularise
            a = torch.cat((inp, cuda_variable(torch.zeros(1, self.embedding_dim, self.n_head) - 111)), 0)
            b = a.unsqueeze(0).repeat(self.num_frames_orchestra, 1, 1, 1)
            c = b.view((self.num_frames_orchestra + 1, self.num_frames_orchestra, self.embedding_dim, self.n_head))
            d = c[:self.num_frames_orchestra]

            # TODO C EST ICI QUIL YA UN TRUC CJHELOU
            #  ICI Les matrices doivent pas être dans le bon sens.
            #  Genre upper triangular au lieu
            #  Par contre pourquoi ???

            # TODO : TESTER  AVEC MATRIX 1 2 3 4  !!!!!!!!!!!!!!

            def expand(mat):
                e = mat.repeat(1, self.num_instruments, 1, 1)
                f = e.view(self.length, self.num_frames_orchestra, self.embedding_dim, self.n_head)
                g = f.permute(1, 0, 2, 3)
                h = g.repeat(1, self.num_instruments, 1, 1)
                g = h.view(self.length, self.length, self.embedding_dim, self.n_head)
                return g

            r_time = expand(d)
            #  Matrices are expanded from permuted inputs... but it's okay, just believe it, it works :)
            if mask is not None:
                mask_expanded = expand(mask).permute(1, 0, 2, 3)
            else:
                mask_expanded = None

            return r_time, mask_expanded

        r_time_past, mask_expanded_past = prepare_r_time(e_past, mask_past)
        if not self.use_masks:
            r_time_future, mask_expanded_future = prepare_r_time(e_future, mask_future)
            # Future self-attention matrices need to be permutted (upper triangular)
            r_time_future_t = r_time_future.permute(1, 0, 2, 3)
            mask_expanded_future_t = mask_expanded_future.permute(1, 0, 2, 3)
            r_time = mask_expanded_past * r_time_past + mask_expanded_future_t * r_time_future_t
        else:
            r_time = r_time_past

        if self.shift:
            r_time = r_time[1:, :-1]

        return r_instrument_repeat, r_time

    def forward(self, q, flipped_masks):
        """

        :param q: (batch_size * n_head, len_q, d)
        :return:
        """
        # TODO use different q for time and instrument
        q_instru = q
        q_time = q
        sz_b = int(q.size()[0] / self.n_head)

        def prepare_r_matrices(r_matrix):
            r_matrix = r_matrix.permute(3, 0, 1, 2)
            # Batch is moving faster in q, so repeat like this is okay here
            r_matrix = r_matrix.repeat(sz_b, 1, 1, 1)
            return r_matrix

        r_instrument_preprocessed, r_time_preprocessed = self.preprocess()
        r_instrument = prepare_r_matrices(r_instrument_preprocessed)
        r_time = prepare_r_matrices(r_time_preprocessed)

        rel_attn_instrument = torch.einsum('bld,blmd->blm', (q_instru, r_instrument))
        rel_attn_time = torch.einsum('bld,blmd->blm', (q_time, r_time))
        rel_attn = rel_attn_instrument + rel_attn_time

        return rel_attn
