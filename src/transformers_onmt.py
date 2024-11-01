from __future__ import division, unicode_literals
import os
import onmt.bin.preprocess as preprocess
import onmt.bin.train as train
import onmt.bin.translate as trsl



class TransformersONMT:
    def __init__(self, 
                 path_to_folder:str='./',
                 dataset_name:str='', 
                 experiment_name:str='', 
                 src_train_path:str='',
                 tgt_train_path:str='', 
                 src_val_path:str='', 
                 tgt_val_path:str='', 
                 src_test_path:str='', 
                 tgt_test_path:str='', 
                 ):
        self.path_to_folder = path_to_folder
        self.dataset_name = dataset_name
        self.experiment_name = experiment_name
        self.src_train_path = src_train_path
        self.tgt_train_path = tgt_train_path
        self.src_val_path = src_val_path
        self.tgt_val_path = tgt_val_path
        self.src_test_path = src_test_path
        self.tgt_test_path = tgt_test_path

        # preprocessing
        self.path_voc = f'{self.path_to_folder}results/transformer_model/{self.dataset_name}/{self.experiment_name}/voc_{self.experiment_name}/'
        # training
        self.path_model_folder = f'{self.path_to_folder}results/transformer_model/{self.dataset_name}/{self.experiment_name}/checkpoints/'
        self.path_logs = f'{self.path_to_folder}results/transformer_model/{self.dataset_name}/{self.experiment_name}/logs/'
        self.tensorboard_log_dir = f'{self.path_to_folder}results/transformer_model/{self.dataset_name}/{self.experiment_name}/tensorboard/'
        # translation
        self.path_pred_output = f'{self.path_to_folder}results/transformer_model/{self.dataset_name}/{self.experiment_name}/predictions/'
        self.output_file_name = f'preds_{self.experiment_name}.txt'

    @staticmethod
    def preprocess_onmt_model(path_src_train:str='', 
                              path_tgt_train:str='', 
                              path_src_val:str='', 
                              path_tgt_val:str='', 
                              folder_path:str='./',
                              path_voc_folder:str='', 
                              src_seq_length:int =3000, 
                              tgt_seq_length:int =3000, 
                              src_vocab_size:int =3000, 
                              tgt_vocab_size:int =3000):
        '''
        Prepare vocabulary for model training. For that, needs to see the validation and training splits of the data to understand how the data is constructed and of which tokens.
        
        --Inputs:
        dataset(str):           name of the dataset at the origin of the data used to prepare the different splits (ex: USPTO_rand_1M)
        experiment (str):       name of the specific experiment done on the splits, usually the name of the model we will be training (such as Tx_mmdd)
        path_to_folder (str):   Path to the folder containing the OpenNMT-py folder itself containing all the data concerning the model we want to work with
        
        --Outputs:
        saves files under ./data/{dataset}/voc_{experiment}/ containing vocabulary needed for the training of the model
        '''
        if path_voc_folder == '':
            path_voc_folder = f'{folder_path}results/transformer_model/voc/'
            print(f'Vocabulary path not specified, saving vocabulary at {path_voc_folder}')

        if not os.path.exists(path_voc_folder):
            os.makedirs(path_voc_folder)

        args = [
            "-train_src",       str(path_src_train),
            "-train_tgt",       str(path_tgt_train),
            "-valid_src",       str(path_src_val),
            "-valid_tgt",       str(path_tgt_val),
            "-save_data",       str(path_voc_folder),
            "-src_seq_length",  str(src_seq_length),
            "-tgt_seq_length",  str(tgt_seq_length), 
            "-src_vocab_size",  str(src_vocab_size), 
            "-tgt_vocab_size",  str(tgt_vocab_size), 
            "-share_vocab",
            "-lower"
        ]

        parser = preprocess._get_parser()
        opt = parser.parse_args(args)
        preprocess.preprocess(opt)

        print(f'Preprocessing completed')

    @staticmethod
    def train_onmt_model(path_voc:str, 
                         path_model_folder:str, 
                         experiment_name:str, 
                         log_folder_path:str,
                         tensorboard_log_dir:str,
                         seed:int =42, 
                         save_checkpoint_steps:int =5000,
                         keep_checkpoints:int =20,
                         train_steps:int =200000, 
                         param_init:float =0,
                         max_generator_batches:int =32,
                         batch_size:int =6144, 
                         batch_type:str ='tokens', 
                         normalization:str ='tokens',
                         max_grad_norm:int =0,
                         accum_count:int =4,
                         optim:str ='adam',
                         adambeta1:float =0.9,
                         adambeta2:float =0.998,
                         decay_method:str ='noam',
                         warmup_steps:int =8000,
                         learnrate:float =2,
                         label_smoothing:float =0.0,
                         layers:int =4,
                         rnn_size:int =384, 
                         word_vec_size:int =384,
                         encoder_type:str ='transformer',
                         decoder_type:str ='transformer',
                         dropout:float =0.1, 
                         global_attention:str ='general',
                         global_attention_function:str ='softmax',
                         self_attn_type:str ='scaled-dot',
                         heads:int =8,
                         transformer_ff:int =2048,
                         valid_steps:int =5000,
                         valid_batch_size:int =4,
                         report_every:int =1000,
                         early_stopping:int =10,
                         early_stopping_criteria:str ='accuracy',
                         world_size:int =1,
                         gpu_ranks:int =0,
                         ):
        '''
        Train transformer model with the onmt package.

        --Inputs
        dataset(str):           name of the dataset at the origin of the data used to prepare the different splits (ex: USPTO_rand_1M)
        experiment (str):       name of the specific experiment done on the splits, usually the name of the model we will be training (such as Tx_mmdd)
        path_to_folder (str):   Path to the folder containing the OpenNMT-py folder itself containing all the data concerning the model we want to work with
        
        --Outputs
        Saves last "-keep_checkpoint" (20 by default) models in .pt files under {path_model_folder}
        Tensorboard files are saved under {tensorboard_log_dir}
        '''

        if not os.path.exists(path_model_folder):
            os.makedirs(path_model_folder)

        if not os.path.exists(log_folder_path):
            os.makedirs(log_folder_path)
        
        if not os.path.exists(tensorboard_log_dir):
            os.makedirs(tensorboard_log_dir)

        args = [
        "-data",                        str(path_voc),
        "-save_model",                  str(path_model_folder),
        "-seed",                        str(seed),
        "-save_checkpoint_steps",       str(save_checkpoint_steps),
        "-keep_checkpoint",             str(keep_checkpoints),
        "-train_steps",                 str(train_steps),
        "-param_init",                  str(param_init),
        "-param_init_glorot",
        "-max_generator_batches",       str(max_generator_batches),
        "-batch_size",                  str(batch_size),
        "-batch_type",                  str(batch_type),
        "-normalization",               str(normalization),
        "-max_grad_norm",               str(max_grad_norm),
        "-accum_count",                 str(accum_count),
        "-optim",                       str(optim),
        "-adam_beta1",                  str(adambeta1),
        "-adam_beta2",                  str(adambeta2),
        "-decay_method",                str(decay_method),
        "-warmup_steps",                str(warmup_steps),
        "-learning_rate",               str(learnrate),
        "-label_smoothing",             str(label_smoothing),
        "-layers",                      str(layers),
        "-rnn_size",                    str(rnn_size),
        "-word_vec_size",               str(word_vec_size),
        "-encoder_type",                str(encoder_type),
        "-decoder_type",                str(decoder_type),
        "-dropout",                     str(dropout),
        "-position_encoding",
        "-global_attention",            str(global_attention),
        "-global_attention_function",   str(global_attention_function),
        "-self_attn_type",              str(self_attn_type),
        "-heads",                       str(heads),
        "-transformer_ff",              str(transformer_ff),
        "-valid_steps",                 str(valid_steps),
        "-valid_batch_size",            str(valid_batch_size),
        "-report_every",                str(report_every),
        "-log_file",                    str(f"{log_folder_path}Training_LOG_{experiment_name}.txt"),
        "-early_stopping",              str(early_stopping),
        "-early_stopping_criteria",     str(early_stopping_criteria),
        "-world_size",                  str(world_size),
        "-gpu_ranks",                   str(gpu_ranks),
        "-tensorboard",
        "-tensorboard_log_dir",         str(tensorboard_log_dir)
        ]

        parser = train._get_parser()
        opt = parser.parse_args(args)
        train.train(opt)
        
        print(f'Training done for experiment {experiment_name}')

    @staticmethod
    def translate_onmt_model(path_model:str, 
                                src_path:str, 
                                output_path:str,
                                output_file_name:str='',  
                                beam_size:int =3, 
                                batch_size:int  = 64, 
                                max_length:int =1000,
                                gpu_ranks:int =0, 
                                ):
        '''
        Model inference.

        --Inputs
        dataset (str):          name of the dataset at the origin of the data used to prepare the different splits (ex: USPTO_rand_1M)
        experiment (str):       name of the specific experiment done on the splits, usually the name of the model we will be training (such as Tx_mmdd)
        path_to_folder (str):   Path to the folder containing the OpenNMT-py folder itself containing all the data concerning the model we want to work with
        step (int):             model step of the model to use for inference 
        src_path (str):         path of the src_test.txt used for inference
        data_inference (str):   name of the dataset at the origin of the src_test.txt for inference
        beam_size (int):        (default 3) number of different predictions the model will make for a given query
        batch_size (int):       (default 64) number of queries performed at the same time for inference

        --Outputs
        List of predictions in txt file (beam_size) x longer as src_test.txt stored under f'{path_to_folder}OpenNMT-py/outputs/{dataset}/{experiment}/output_{experiment}_{step}.txt'
        '''

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        args = [
            "-beam_size",   str(beam_size), 
            "-n_best",      str(beam_size), 
            "-model",       str(path_model), 
            "-src",         str(src_path), 
            "-output",      f'{output_path}{output_file_name}', 
            "-batch_size",  str(batch_size), 
            "-max_length",  str(max_length), 
            "-gpu",         str(gpu_ranks), 
            "-log_probs",
            "-replace_unk"
        ]

        parser = trsl._get_parser()
        opt = parser.parse_args(args)
        trsl.translate(opt)


    def preprocess(self, **kwargs):
        '''
        Perform preprocessing of the data for the class instance
        '''
        self.preprocess_onmt_model(
            path_src_train=self.src_train_path, 
            path_tgt_train=self.tgt_train_path, 
            path_src_val=self.src_val_path, 
            path_tgt_val=self.tgt_val_path, 
            folder_path=self.path_to_folder,
            path_voc_folder=self.path_voc, 
            **kwargs
            )
    
    def train(self, **kwargs):
        '''
        Train model for the class instance
        '''
        self.train_onmt_model(
            path_voc=self.path_voc, 
            path_model_folder=self.path_model_folder, 
            experiment_name=self.experiment_name, 
            log_folder_path=self.path_logs,
            tensorboard_log_dir=self.tensorboard_log_dir, 
            **kwargs
            )
    
    def translate(self, path_model, **kwargs):
        '''
        Translate data for the class instance
        '''
        self.translate_onmt_model(
            path_model=path_model,  
            src_path=self.src_test_path, 
            output_path=self.path_pred_output, 
            output_file_name=self.output_file_name, 
            **kwargs
            )