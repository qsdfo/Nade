import math
import os
import random
from itertools import islice
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F

from Nade.layers import FFNN


class Nade_mlp(nn.Module):
    def __init__(self,
                 dataset,
                 data_processor,
                 layers_condition,
                 layers_NADE,
                 layers_last_module,
                 dropout,
                 input_dropout,
                 lr=1e-3,
                 gpu_ids=[0]
                 ):
        super(Nade_mlp, self).__init__()

        self.data_processor = data_processor

        # Condition
        cond_input_dim = [data_processor.concatenated_embedded_inputs_dim] + layers_condition
        self.condition_encoder = FFNN(cond_input_dim, dropout)
        # Nade part
        orchestra_dim = [data_processor.orchestra_dim] + layers_NADE
        self.nade = FFNN(orchestra_dim, dropout)
        # Last part
        last_module_dim = [layers_NADE[-1] + layers_condition[-1]] + layers_last_module
        self.prediction_module = FFNN(last_module_dim, dropout)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def __repr__(self):
        name = "Nade_mlp"
        return f'{name}_{self.dataset.__repr__()}'

    @property
    def model_dir(self):
        return f'models/{self.__repr__()}'

    @property
    def log_dir(self):
        log_dir = f'logs/{self.__repr__()}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.state_dict(), f'{self.model_dir}/state_dict')
        print(f'Model {self.__repr__()} saved')

    def save_overfit(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.state_dict(), f'{self.model_dir}/state_dict_overfit')
        print(f'Overfitted model {self.__repr__()} saved')

    def load(self):
        print(f'Loading model {self.__repr__()}')
        self.load_state_dict(torch.load(f'{self.model_dir}/state_dict'))

    def load_overfit(self):
        print(f'Loading overfitted model {self.__repr__()}')
        self.load_state_dict(torch.load(f'{self.model_dir}/state_dict_overfit'))

    def forward(self, piano, orchestra_past, orchestra_t, mask, exclude_rests=False):
        # Embed piano
        p_embed = self.data_processor.embed_piano(piano)
        # Embed orchestra
        o_past_embed = self.data_processor.embed_orchestra_past(orchestra_past)
        # Concatenate them
        condition = torch.cat((p_embed, o_past_embed), dim=1)
        # Encode
        condition_encoded = self.condition_encoder(condition)

        # Nade
        # apply binary mask
        masked_o_t = mask * orchestra_t
        # concatenate mask
        nade_input = torch.cat((orchestra_t, masked_o_t))
        nade_encoded = self.nade(nade_input)

        # Concatenate both output
        torch.cat((condition_encoded, nade_encoded), dim=1)

        # 


        if self.conditioning:
            # No masking on encoder's inputs
            # Todo But perhaps use dropout_input here ?
            enc_output, *_ = self.encoder(x=x_enc,
                                          enc_outputs=None,
                                          return_attns=False,
                                          return_all_layers=self.hierarchical_encoding)
        else:
            enc_output = None

        # Random padding of decoder input when nade
        if self.nade:
            masked_x_dec, mask_padding = self.data_processor_decoder.mask(x_dec, p=None)
        else:
            masked_x_dec = x_dec

        pred_seq, *_ = self.decoder.forward(x=masked_x_dec,
                                            enc_outputs=enc_output,
                                            return_attns=False,
                                            return_all_layers=False,
                                            )

        preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)

        if not self.nade:
            mask_padding = None
        loss = self.data_processor_decoder.mean_crossentropy(preds=preds,
                                                             targets=x_dec,
                                                             mask=mask_padding,
                                                             shift=(not self.nade))

        samples = None
        return {'loss': loss,
                'monitored_quantities': {'loss': loss.mean().item()},
                'samples': samples
                }

    def train_model(self,
                    batch_size,
                    num_batches=None,
                    num_samples=1,
                    num_epochs=10,
                    plot=False,
                    **kwargs
                    ):

        (generator_train,
         generator_val,
         generator_test) = self.dataset.data_loaders(batch_size=batch_size)

        if plot:
            import tensorboardX
            from tensorboardX import SummaryWriter

        best_validation = math.inf

        for epoch_id in range(num_epochs):
            monitored_quantities_train = self.epoch(
                data_loader=generator_train,
                train=True,
                num_batches=num_batches,
            )

            monitored_quantities_val = self.epoch(
                data_loader=generator_val,
                train=False,
                num_batches=num_batches // 2 if num_batches is not None else None,
            )

            print(f'======= Epoch {epoch_id} =======')
            print(f'---Train---')
            dict_pretty_print(monitored_quantities_train, endstr=' ' * 5)
            print()
            print(f'---Val---')
            dict_pretty_print(monitored_quantities_val, endstr=' ' * 5)
            print('\n')

            if monitored_quantities_val["loss"] < best_validation:
                print(f'Saving model!')
                self.save()
                best_validation = monitored_quantities_val["loss"]
            #  Also save overfitted
            self.save_overfit()

    def epoch(self, data_loader,
              train=True,
              num_batches=None,
              ):
        if num_batches is None or num_batches > len(data_loader):
            num_batches = len(data_loader)

        means = None

        if train:
            self.train()
        else:
            self.eval()

        for sample_id, tensors in tqdm(enumerate(islice(data_loader,
                                                        num_batches))):

            x_enc = self.data_processor_encoder.preprocessing(*tensors)
            x_dec = self.data_processor_decoder.preprocessing(*tensors)

            self.optimizer.zero_grad()
            forward_pass_gen = self.forward(x_enc, x_dec)
            loss = forward_pass_gen['loss']

            if train:
                loss.backward()
                # Todo clip grad?!
                # torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                self.optimizer.step()

            # Monitored quantities
            monitored_quantities = dict(forward_pass_gen['monitored_quantities'])
            # average quantities
            if means is None:
                means = {key: 0
                         for key in monitored_quantities}
            means = {
                key: value + means[key]
                for key, value in monitored_quantities.items()
            }

            del forward_pass_gen
            del loss

        # Re-normalize monitored quantities
        means = {
            key: value / num_batches
            for key, value in means.items()
        }

        return means

    # def generation_from_ascii(self, ascii_melody=None):
    #     self.eval()
    #     batch_size = 1
    #     num_measures = len(ascii_melody) // 4 // 4
    #     sequences_size = self.dataset.sequences_size
    #     assert sequences_size % 2 == 0
    #     # constraints_size = sequences_size // 2 - 1
    #     constraints_size = sequences_size // 2
    #     # constraints_size = sequences_size // 2 + 1
    #     context_size = sequences_size - constraints_size
    #     subdivision = self.dataset.subdivision
    #
    #     with torch.no_grad():
    #         chorale, constraint_chorale = self.init_generation(num_measures,
    #                                                            ascii_melody=ascii_melody)
    #         for beat_index in range(4 * (num_measures - context_size)):
    #             # iterations per beat
    #             # mandatory -> position
    #
    #             for tick_index in range(subdivision):
    #                 time_index = beat_index * 4 + tick_index
    #                 remaining_tick_index = 3 - tick_index
    #
    #                 if time_index % 16 == 0:
    #                     # exclude_symbols = ['START', 'END', '__']
    #                     exclude_symbols = ['START', 'END']
    #                 else:
    #                     exclude_symbols = ['START', 'END']
    #
    #                 for voice_index in range(4):
    #                     enc_output, *_ = self.encoder.forward(
    #                         x=constraint_chorale[:, :,
    #                                 beat_index * subdivision:(beat_index +
    #                                                           sequences_size) * subdivision],
    #                         enc_output=None,
    #                         shift=False
    #                     )
    #                     pred_seq, *_ = self.decoder.forward(
    #                         x=chorale[:, :, beat_index * subdivision:(beat_index +
    #                                                                   sequences_size) * subdivision],
    #                         enc_output=enc_output
    #                     )
    #                     preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)
    #
    #                     probs = F.softmax(
    #                         preds[voice_index][:,
    #                         -1 - remaining_tick_index - constraints_size * subdivision,
    #                         :],
    #                         dim=1)
    #
    #                     p = to_numpy(probs[0])
    #                     # temperature ?!
    #                     p = np.exp(np.log(p) * 1.2)
    #
    #                     # exclude non note symbols:
    #                     for sym in exclude_symbols:
    #                         sym_index = self.dataset.note2index_dicts[voice_index][sym]
    #                         p[sym_index] = 0
    #
    #                     p = p / sum(p)
    #                     new_pitch_index = np.random.choice(np.arange(
    #                         self.data_processor_decoder.num_notes_per_voice[voice_index]
    #                     ), p=p)
    #                     # new_pitch_index = np.argmax(p)
    #                     chorale[:, voice_index,
    #                     beat_index * subdivision + context_size * subdivision - remaining_tick_index - 1] = \
    #                         int(
    #                             new_pitch_index)
    #                     # constraint_chorale[:, voice_index,
    #                     # beat_index * subdivision + context_size * subdivision -
    #                     # remaining_tick_index - 1] = int(new_pitch_index)
    #
    #     tensor_score = chorale
    #     score = self.dataset.tensor_to_score(
    #         tensor_score[0]
    #     )
    #     return score, tensor_score, None

    # def generation(self, ascii_melody=None):
    #     self.eval()
    #     batch_size = 1
    #     num_measures = 16
    #
    #     sequences_size = self.dataset.sequences_size
    #     assert sequences_size % 2 == 0
    #     # constraints_size = sequences_size // 2 - 1
    #     constraints_size = sequences_size // 2
    #     # constraints_size = sequences_size // 2 + 1
    #     context_size = sequences_size - constraints_size
    #     subdivision = self.dataset.subdivision
    #
    #     with torch.no_grad():
    #         chorale, constraint_chorale = self.init_generation(num_measures,
    #                                                            ascii_melody=ascii_melody)
    #         for beat_index in range(4 * (num_measures - context_size)):
    #             # iterations per beat
    #             # mandatory -> position
    #
    #             for tick_index in range(subdivision):
    #                 time_index = beat_index * 4 + tick_index
    #                 remaining_tick_index = 3 - tick_index
    #
    #                 if time_index % 16 == 0:
    #                     # exclude_symbols = ['START', 'END', '__']
    #                     exclude_symbols = ['START', 'END']
    #                 else:
    #                     exclude_symbols = ['START', 'END']
    #
    #                 for voice_index in range(4):
    #                     enc_output, *_ = self.encoder.forward(
    #                         x=constraint_chorale[:, :,
    #                                 beat_index * subdivision:(beat_index +
    #                                                           sequences_size) * subdivision],
    #                         shift=False
    #                     )
    #                     pred_seq, *_ = self.decoder.forward(
    #                         x=chorale[:, :, beat_index * subdivision:(beat_index +
    #                                                                   sequences_size) * subdivision],
    #                         enc_output=enc_output
    #                     )
    #                     preds = self.pred_seq_to_preds(pred_seq)
    #
    #                     probs = F.softmax(
    #                         preds[voice_index][:,
    #                         -1 - remaining_tick_index - constraints_size * subdivision,
    #                         :],
    #                         dim=1)
    #
    #                     p = to_numpy(probs[0])
    #                     # temperature ?!
    #                     p = np.exp(np.log(p) * 1.2)
    #
    #                     # exclude non note symbols:
    #                     for sym in exclude_symbols:
    #                         sym_index = self.dataset.note2index_dicts[voice_index][sym]
    #                         p[sym_index] = 0
    #
    #                     p = p / sum(p)
    #                     new_pitch_index = np.random.choice(np.arange(
    #                         self.num_notes_per_voice[voice_index]
    #                     ), p=p)
    #                     # new_pitch_index = np.argmax(p)
    #                     chorale[:, voice_index,
    #                     beat_index * subdivision + context_size * subdivision - remaining_tick_index - 1] = \
    #                         int(
    #                             new_pitch_index)
    #                     # constraint_chorale[:, voice_index,
    #                     # beat_index * subdivision + context_size * subdivision -
    #                     # remaining_tick_index - 1] = int(new_pitch_index)
    #
    #     tensor_score = chorale
    #     score = self.dataset.tensor_to_score(
    #         tensor_score[0]
    #     )
    #     return score, tensor_score, None

    def unconstrained_generation(self, ascii_melody=None,
                                 num_beats=16,
                                 # todo this parameter should be in dataset
                                 num_tokens_per_beat=None):
        self.eval()
        print('WARNING: works only with LSDB')
        batch_size = 1

        sequences_size = self.dataset.sequences_size
        assert sequences_size % 2 == 0
        # constraints_size = sequences_size // 2 - 1
        constraints_size = 8
        # constraints_size = sequences_size // 2 + 1
        num_beats_context = sequences_size - constraints_size
        num_beats_constraints = constraints_size

        exclude_symbols = ['START', 'END']

        with torch.no_grad():
            x_flatten, x_constraints_flatten = self.data_processor_encoder.init_generation(
                num_beats,
                num_beats_context,
                num_beats_constraints)

            for beat_index in range(num_beats):
                # iterations per beat
                # mandatory -> position

                # tick_index is the index of the subdivision of the current beat
                for tick_index in range(num_tokens_per_beat):
                    time_index_in_slice = num_beats_context * num_tokens_per_beat + tick_index

                    x_slice = self.data_processor_encoder.wrap(
                        x_flatten[:, beat_index * num_tokens_per_beat:
                                     (beat_index + sequences_size) * num_tokens_per_beat])

                    pred_seq, *_ = self.decoder.forward(
                        x=x_slice,
                        enc_output=None,
                        shift=True,
                        embed=True
                    )
                    preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)

                    # pad to easily implement softmax
                    flattened_probabilities = (
                        self.data_processor_decoder.preds_to_flattened_probabilities(preds)
                    )

                    probs = F.softmax(flattened_probabilities, dim=2)

                    p = to_numpy(probs[0])

                    p = p[time_index_in_slice]
                    # temperature ?!
                    p = np.exp(np.log(p) * 0.9)

                    # exclude non note symbols:
                    for sym in exclude_symbols:
                        # Todo Unification
                        # symbol2index_dicts not called like that in BachChorales
                        # Dataset
                        sym_index = self.dataset.symbol2index_dicts[
                            self.data_processor_encoder.tick_index_to_dict_index(
                                time_index_in_slice)
                        ][sym]
                        p[sym_index] = 0
                    p = p / sum(p)

                    new_pitch_index = np.random.choice(np.arange(
                        len(p)
                    ), p=p)
                    # new_pitch_index = np.argmax(p)
                    # TODO check indexes
                    x_flatten[:,
                    beat_index * num_tokens_per_beat + time_index_in_slice] = \
                        int(new_pitch_index)
                    x_constraints_flatten[:,
                    beat_index * num_tokens_per_beat + time_index_in_slice] = \
                        int(new_pitch_index)

        # remove start symbols
        x_flatten = x_flatten[:,
                    num_beats_context * num_tokens_per_beat:
                    -num_beats_constraints *
                    num_tokens_per_beat]
        # TODO! Dependent of the dataset :(
        tensor_score = self.data_processor_encoder.wrap(x_flatten)
        tensor_score = [t.detach().cpu()
                        for t in tensor_score]
        score = self.dataset.tensor_to_score(
            tensor_score[0], (tensor_score[1], tensor_score[2]),
            realize_chords=True,
            add_chord_symbols=True

        )
        return score, tensor_score, None

    def generation(self, ascii_melody=None,
                   num_beats=16,
                   # todo this parameter should be in dataset
                   num_tokens_per_beat=None,
                   temperature=1.,
                   ):
        self.eval()
        print('WARNING: works only with LSDB')
        batch_size = 1

        sequences_size = self.dataset.sequences_size
        assert sequences_size % 2 == 0
        # constraints_size = sequences_size // 2 - 1
        # constraints_size = sequences_size // 2
        constraints_size = sequences_size // 2 + 1
        # constraints_size = 8
        num_beats_context = sequences_size - constraints_size
        num_beats_constraints = constraints_size

        exclude_symbols = ['START', 'END', 'XX']

        with torch.no_grad():
            x_flatten, x_constraints_flatten = self.data_processor_encoder.init_generation(
                num_beats,
                num_beats_context,
                num_beats_constraints)

            for beat_index in range(num_beats):
                # iterations per beat
                # mandatory -> position

                # tick_index is the index of the subdivision of the current beat
                for tick_index in range(num_tokens_per_beat):
                    time_index_in_slice = num_beats_context * num_tokens_per_beat + tick_index

                    x_constraints_slice = self.data_processor_encoder.wrap(
                        x_constraints_flatten[:,
                        beat_index * num_tokens_per_beat:
                        (beat_index + sequences_size) * num_tokens_per_beat])

                    if self.conditioning:
                        enc_output, *_ = self.encoder.forward(
                            x=x_constraints_slice,
                            enc_output=None,
                            shift=False,
                            embed=True
                        )
                    else:
                        enc_output = None

                    x_slice = self.data_processor_encoder.wrap(
                        x_flatten[:, beat_index * num_tokens_per_beat:
                                     (beat_index + sequences_size) * num_tokens_per_beat])

                    pred_seq, *_ = self.decoder.forward(
                        x=x_slice,
                        enc_output=enc_output,
                        shift=True,
                        embed=True
                    )
                    preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)

                    # pad to easily implement softmax
                    flattened_probabilities = (
                        self.data_processor_decoder.preds_to_flattened_probabilities(preds)
                    )

                    probs = F.softmax(flattened_probabilities, dim=2)

                    p = to_numpy(probs[0])

                    p = p[time_index_in_slice]
                    # temperature ?!
                    p = np.exp(np.log(p) * temperature)

                    # exclude non note symbols:
                    for sym in exclude_symbols:
                        # Todo Unification
                        # symbol2index_dicts not called like that in BachChorales
                        # Dataset
                        sym_index = self.dataset.symbol2index_dicts[
                            self.data_processor_encoder.tick_index_to_dict_index(
                                time_index_in_slice)
                        ][sym]
                        p[sym_index] = 0
                    p = p / sum(p)

                    new_pitch_index = np.random.choice(np.arange(
                        len(p)
                    ), p=p)
                    # new_pitch_index = np.argmax(p)
                    # TODO check indexes
                    x_flatten[:,
                    beat_index * num_tokens_per_beat + time_index_in_slice] = \
                        int(new_pitch_index)
                    x_constraints_flatten[:,
                    beat_index * num_tokens_per_beat + time_index_in_slice] = \
                        int(new_pitch_index)

        # remove start symbols
        x_flatten = x_flatten[:,
                    num_beats_context * num_tokens_per_beat:
                    -num_beats_constraints *
                    num_tokens_per_beat]
        # TODO! Dependent of the dataset :(
        tensor_score = self.data_processor_encoder.wrap(x_flatten)
        tensor_score = [t.detach().cpu()
                        for t in tensor_score]
        score = self.dataset.tensor_to_score(
            tensor_score[0], (tensor_score[1], tensor_score[2]),
            realize_chords=True,
            add_chord_symbols=True

        )
        return score, tensor_score, None

    def generation_arrangement_frame(self,
                                     temperature=1.,
                                     filepath=None,
                                     banned_instruments=[]
                                     ):
        self.eval()
        print('WARNING: works only with Arrangement')

        with torch.no_grad():

            if filepath:
                piano, rhythm_piano, orchestra, orchestra_silenced_instruments = \
                    self.data_processor_encoder.init_generation_filepath(filepath, banned_instruments)
            else:
                piano, rhythm_piano, orchestra, orchestra_silenced_instruments = \
                    self.data_processor_encoder.init_generation(banned_instruments=banned_instruments)

            if self.conditioning:
                enc_output, *_ = self.encoder.forward(
                    x=piano,
                    enc_outputs=None,
                    shift=False,
                    embed=True
                )
            else:
                raise Exception("Arrangement needs a conditioning")

            for instrument_index in range(self.data_processor_decoder.num_instruments):

                if orchestra_silenced_instruments[instrument_index] == 1:
                    continue

                pred_seq, *_ = self.decoder.forward(
                    x=orchestra,
                    enc_output=enc_output,
                    shift=True,
                    embed=True
                )
                preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)
                pred = preds[instrument_index]
                # todo DO NOT USE THE UPDATE ORCHESTRA WITH THE GENERATED SAMPLES FOR KNOWN FIXED INSTRUMENTS (SEE MASK RETURNED BY

                prob = F.softmax(pred, dim=1)
                p = to_numpy(prob)
                # temperature ?!
                p = np.exp(np.log(p) * temperature)
                p = p / np.sum(p, axis=1, keepdims=True)
                batch_size = len(p)

                for batch_index in range(batch_size):
                    # new_pitch_index = np.argmax(p)
                    new_pitch = np.random.choice(np.arange(len(p[0])), p=p[batch_index])
                    orchestra[batch_index, instrument_index] = int(new_pitch)

        # Force silences in orchestra (not trained on silent piano frames...)
        silences_in_piano = (piano.sum(dim=1) == 0)
        for frame_index, is_silence in enumerate(silences_in_piano):
            if is_silence == 1:
                orchestra[frame_index, :] = 0

        piano_cpu = piano.cpu()
        orchestra_cpu = orchestra.cpu()
        self.dataset.visualise_batch((piano_cpu, rhythm_piano), (orchestra_cpu, rhythm_piano), writing_dir=self.log_dir,
                                     filepath="generation")
        return

    def generation_arrangement(self,
                               temperature=1.,
                               batch_size=2,
                               filepath=None,
                               write_name=None,
                               banned_instruments=[],
                               plot_attentions=False,
                               ):
        # WARNING: works only with Arrangement

        self.eval()

        # Number of complete pass over all time frames in, non auto-regressive sampling schemes
        number_sampling_steps = 5

        print(f'# {filepath}')

        with torch.no_grad():

            if filepath:
                piano, rhythm_piano, orchestra, orchestra_silenced_instruments = \
                    self.data_processor_encoder.init_generation_filepath(batch_size, filepath, banned_instruments)
            else:
                piano, rhythm_piano, orchestra, orchestra_silenced_instruments = \
                    self.data_processor_encoder.init_generation(banned_instruments=banned_instruments)

            context_size = self.data_processor_decoder.num_frames_orchestra - 1
            first_frame = context_size
            last_frame = piano.size()[1] - 1 - context_size
            if self.nade:
                time_frames = list(range(first_frame, last_frame + 1))
                time_frames *= number_sampling_steps
                random.shuffle(time_frames)
            else:
                time_frames = list(range(first_frame, last_frame + 1))
            for frame_index in time_frames:
                # Get context
                start_piano = frame_index - context_size
                end_piano = frame_index + context_size
                piano_context = piano[:, start_piano:end_piano + 1, :]

                x_enc = self.data_processor_encoder.preprocessing(piano_context, None)

                return_encoder = self.encoder.forward(
                    x=x_enc,
                    enc_outputs=None,
                    return_attns=plot_attentions,
                    return_all_layers=self.hierarchical_encoding,
                    embed=True,
                )

                if plot_attentions:
                    enc_outputs, enc_slf_attn, _ = return_encoder
                else:
                    enc_outputs, = return_encoder

                #  Sample sequentially each instrument
                for instrument_index in range(self.data_processor_decoder.num_instruments):

                    if orchestra_silenced_instruments[instrument_index] == 1:
                        continue

                    orchestra_context = orchestra[:, start_piano:end_piano + 1, :]
                    x_dec = self.data_processor_decoder.preprocessing(None, orchestra_context)

                    return_decoder = self.decoder.forward(
                        x=x_dec,
                        enc_outputs=enc_outputs,
                        embed=True,
                        return_attns=plot_attentions
                    )

                    if plot_attentions:
                        pred_seq, dec_slf_attn, dec_enc_attn = return_decoder
                    else:
                        pred_seq, = return_decoder

                    preds = self.data_processor_decoder.pred_seq_to_preds(pred_seq)
                    pred = preds[instrument_index]
                    # Prediction is in the last frame
                    pred_t = pred[:, -1, :]

                    prob = F.softmax(pred_t, dim=1)
                    p = to_numpy(prob)
                    # temperature ?!
                    p = np.exp(np.log(p) * temperature)
                    p = p / np.sum(p, axis=1, keepdims=True)
                    batch_size = len(p)

                    for batch_index in range(batch_size):
                        # new_pitch_index = np.argmax(p)
                        predicted_one_hot_value = np.random.choice(np.arange(len(p[0])), p=p[batch_index])
                        predicted_symbol = self.data_processor_decoder.dataset.index2midi_pitch[instrument_index][
                            predicted_one_hot_value]
                        if predicted_symbol == START_SYMBOL:
                            print("START")
                        elif predicted_symbol == END_SYMBOL:
                            print("END")
                        elif predicted_symbol == PAD_SYMBOL:
                            print("PAD")
                        orchestra[batch_index, frame_index, instrument_index] = int(predicted_one_hot_value)

        # Force silences in orchestra (not trained on silent piano frames...)
        # silences_in_piano = (piano[0].sum(dim=-1) == 0)
        # for frame_index, is_silence in enumerate(silences_in_piano):
        #     if is_silence == 1:
        #         orchestra[:, frame_index, :] = 0

        piano_cpu = piano[:, context_size:-context_size].cpu()
        orchestra_cpu = orchestra[:, context_size:-context_size].cpu()
        # Last duration will be a quarter length
        duration_piano = np.asarray(rhythm_piano[1:] + [self.dataset.subdivision]) - np.asarray(rhythm_piano[:-1] + [0])
        for batch_index in range(batch_size):
            self.dataset.visualise_batch(piano_cpu[batch_index], orchestra_cpu[batch_index], duration_piano,
                                         writing_dir=self.log_dir, filepath=f"{write_name}_{batch_index}")
        return


def dict_pretty_print(d, endstr='\n'):
    for key, value in d.items():
        print(f'{key.capitalize()}: {value:.6}', end=endstr)
