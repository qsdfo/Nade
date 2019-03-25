# TODO
#  Scheduling for input masking in NADE-like models

import torch

import click
from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
import os

from DatasetManager.dataset_manager import DatasetManager

print(os.getcwd())

from Nade.arrangement.arrangement_data_processor import ArrangementDataProcessor
from Nade.config import get_config


@click.command()
@click.option('--num_layers', default=2,
              help='number of layers of the LSTMs')
@click.option('--dropout', default=0.1,
              help='amount of dropout between layers')
@click.option('--input_dropout', default=0.2,
              help='amount of dropout on input')
@click.option('--lr', default=1e-4,
              help='learning rate')
@click.option('--per_head_dim', default=64,
              help='Feature dimension in each head')
@click.option('--num_heads', default=8,
              help='Number of heads')
@click.option('--local_position_embedding_dim', default=8,
              help='Embedding size for local positions')
@click.option('--position_ff_dim', default=1024,
              help='Hidden dimension of the position-wise ffnn')
#
@click.option('--batch_size', default=128,
              help='training batch size')
@click.option('--num_batches', default=None, type=int,
              help='Number of batches per epoch, None for all dataset')
@click.option('--num_epochs', default=100,
              help='number of training epochs')
@click.option('--action', type=click.Choice(['train', 'load', 'load_overfit']),
              help='Choose the action to perform (train, load, load_overfit...)')
#
@click.option('--plot_attentions', is_flag=True,
              help='Plot attentions')
@click.option('--conditioning', is_flag=True,
              help='condition on set of constraints')
@click.option('--dataset_type', type=click.Choice(['arrangement', 'arrangement_test']))
@click.option('--model_type',
              type=click.Choice(['transformer', 'transformer_nade', 'transformer_hierarchical']),
              default='transformer')
@click.option('--block_attention', is_flag=True,
              help='Do we use block attention ?')
#
@click.option('--num_examples_sampled', default=5,
              help='number of orchestration generated per given piano input')
@click.option('--single_gpu', is_flag=True,
              help='Flag for using only one gpu (for debuging)')
def main(block_attention,
         num_layers,
         dropout,
         input_dropout,
         lr,
         per_head_dim,
         num_heads,
         local_position_embedding_dim,
         position_ff_dim,
         batch_size,
         num_epochs,
         action,
         plot_attentions,
         num_batches,
         dataset_type,
         model_type,
         conditioning,
         num_examples_sampled,
         single_gpu
         ):
    # Use all gpus available
    if single_gpu:
        gpu_ids = [0]
    else:
        gpu_ids = [int(gpu) for gpu in range(torch.cuda.device_count())]

    config = get_config()

    if dataset_type == 'arrangement':
        dataset_manager = DatasetManager()
        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': 2,
            'sequence_size': 3,
            'velocity_quantization': 2,
            'max_transposition': 6,
            'compute_statistics_flag': False
        }
        dataset: ArrangementDataset = dataset_manager.get_dataset(
            name='arrangement',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ArrangementDataProcessor(dataset=dataset,
                                                     embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                     reducer_input_dim=reducer_input_dim,
                                                     local_position_embedding_dim=local_position_embedding_dim,
                                                     flag_orchestra=False,
                                                     block_attention=False)

        processor_decoder = ArrangementDataProcessor(dataset=dataset,
                                                     embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                     reducer_input_dim=reducer_input_dim,
                                                     local_position_embedding_dim=local_position_embedding_dim,
                                                     flag_orchestra=True,
                                                     block_attention=block_attention)

    elif dataset_type == 'arrangement_test':
        dataset_manager = DatasetManager()
        arrangement_dataset_kwargs = {
            'transpose_to_sounding_pitch': True,
            'subdivision': 2,
            'sequence_size': 3,
            'velocity_quantization': 2,
            'max_transposition': 6,
            'compute_statistics_flag': False
        }
        dataset: ArrangementDataset = dataset_manager.get_dataset(
            name='arrangement_test',
            **arrangement_dataset_kwargs
        )

        reducer_input_dim = num_heads * per_head_dim

        processor_encoder = ArrangementDataProcessor(dataset=dataset,
                                                     embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                     reducer_input_dim=reducer_input_dim,
                                                     local_position_embedding_dim=local_position_embedding_dim,
                                                     flag_orchestra=False,
                                                     block_attention=False)

        processor_decoder = ArrangementDataProcessor(dataset=dataset,
                                                     embedding_dim=reducer_input_dim - local_position_embedding_dim,
                                                     reducer_input_dim=reducer_input_dim,
                                                     local_position_embedding_dim=local_position_embedding_dim,
                                                     flag_orchestra=True,
                                                     block_attention=block_attention)

    # elif dataset_type == 'arrangement_frame':
    #     dataset_manager = DatasetManager()
    #     arrangement_dataset_kwargs = {
    #         'transpose_to_sounding_pitch': True,
    #         'subdivision': 2,
    #         'compute_statistics_flag': "/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/statistics"
    #     }
    #     dataset: ArrangementFrameDataset = dataset_manager.get_dataset(
    #         name='arrangement_frame',
    #         **arrangement_dataset_kwargs
    #     )
    #
    #     processor_encoder = ArrangementFrameDataProcessor(dataset=dataset,
    #                                                       embedding_dim=512 - 8,
    #                                                       reducer_input_dim=512,
    #                                                       local_position_embedding_dim=8,
    #                                                       flag_orchestra=False)
    #
    #     processor_decoder = ArrangementFrameDataProcessor(dataset=dataset,
    #                                                       embedding_dim=512 - 8,
    #                                                       reducer_input_dim=512,
    #                                                       local_position_embedding_dim=8,
    #                                                       flag_orchestra=True)

    else:
        raise NotImplementedError
    # Todo Nade_rnn
    if model_type == 'Nade_mlp':
        model = Nade_mlp(dataset=dataset,
                            data_processor_encoder=processor_encoder,
                            data_processor_decoder=processor_decoder,
                            num_heads=num_heads,
                            per_head_dim=per_head_dim,
                            position_ff_dim=position_ff_dim,
                            hierarchical_encoding=False,
                            block_attention=block_attention,
                            nade=False,
                            num_layers=num_layers,
                            dropout=dropout,
                            input_dropout=input_dropout,
                            conditioning=conditioning,
                            lr=lr,
                            gpu_ids=gpu_ids
                            )

    if action == 'load':
        model.load()
    elif action == 'load_overfit':
        model.load_overfit()

    model.cuda()

    if action == 'train':
        print(f"Train the model on gpus {gpu_ids}")
        model.train_model(batch_size=batch_size,
                          num_epochs=num_epochs,
                          num_batches=num_batches,
                          plot=True)

    print('Generation')
    temperature = 1.
    # banned_instruments = ["Violin_1", "Violin_2"]
    banned_instruments = []
    source_folder = f"{config['datapath']}/source_for_generation/"
    filepaths = [
        (source_folder + "Moussorgsky_TableauxProm(24 mes)_solo.xml", "mouss_picturesExhib"),
        # (source_folder + "Debussy_SuiteBergam_Prelude(89 mes).xml", "Debussy_SuiteBergam_Prelude"),
        # (source_folder + "Mozart_KlNachtmusik_i(1-55).xml", "Mozart_KlNachtmusik"),
        # (source_folder + "Tchaikovsky_Symph4_i_Intro(1-26).xml", "Tchaikovsky_Symph4"),
    ]
    for filepath in filepaths:
        model.generation_arrangement(
            temperature=temperature,
            batch_size=num_examples_sampled,
            filepath=filepath[0],
            write_name=filepath[1],
            banned_instruments=banned_instruments,
            plot_attentions=plot_attentions)


if __name__ == '__main__':
    main()
