# Easy training and inference of OpenNMT-py transformer models

With this package, you can simply train and use OpenNMT-py transformer models (version 1.1.1, older but with confidence score attributed to predictions)

## Package installation
### From GitHub
```
git clone git@github.com:yvsgrndjn/transformers_onmt.git
cd transformers_onmt
conda create -n transformers_onmt python=3.8.16 -y
conda activate transformers_onmt
pip install -e .
```
## Project structure
Once instantiated, the class will reproduce this file structure:
```
results/
├── transformer_model/
|	├── {dataset_name}
|	| 	├── {experiment_name}
|	| 	|	├── checkpoints/ # where the models are saved during training
|	| 	|	├── Tensorboard/ # stores files allowing to analyze models
|	| 	|	├── logs/ 	 # logs on training
|	| 	|	├── voc_{experiment_name} # where vocabs are stored under .files
|	| 	|	├── predictions/ # predictions are stored here
|	└── ..
└──(in case we are dealing with other models)
```
## Usage
### Using models without depending on class instantiation
In the case you are in the `notebooks` folder trying this on a notebook for example.
Preprocessing allows to build the vocabulary the model will later use during training:
```
#import the package
from transformers_onmt import TransformersONMT

# define the paths of your source and target .txt files for each split
src_train_path_1000 = '../examples/src_train_1000first.txt'
tgt_train_path_1000 = '../examples/tgt_train_1000first.txt'

src_val_path_1000   = '../examples/src_val_1000first.txt'
tgt_val_path_1000   = '../examples/tgt_val_1000first.txt'

src_test_path_1000  = '../examples/src_test_1000first.txt'
tgt_test_path_1000  = '../examples/tgt_test_1000first.txt'

TransformersONMT.preprocess_onmt_model(
    path_src_train=src_train_path_1000, 
    path_tgt_train=tgt_train_path_1000, 
    path_src_val=src_val_path_1000, 
    path_tgt_val=tgt_val_path_1000, 
    folder_path='../'
)
```
For training a model with the necessary arguments (please take a look at all the keyboard arguments that you can modify to taylor it to your specific needs, here for example we set the optional argument
 number of training steps `train_steps= 1000`)
```
TransformersONMT.train_onmt_model(
    path_voc='../results/transformer_model/voc/', 
    path_model_folder=f'../results/transformer_model/checkpoints/', 
    experiment_name='test', 
    log_folder_path='../results/transformer_model/logs/', 
    tensorboard_log_dir='../results/transformer_model/tensorboard_logs/',
    train_steps=1000, 
)
```
For model inference, we use: (again, there are optional keyboard arguments you can input here)
```
TransformersONMT.translate_onmt_model(
    path_model='../results/transformer_model/checkpoints/model_test.pt', 
    src_path=src_test_path_1000, 
    output_path='../results/transformer_model/predictions/', 
    output_file_name='test.txt'

)
```
### Using the package with class instantiation
Here we will let the package sort the files as shown in the project structure. The instantiation can be performed as follows:
```
# import the class from the package
from transformers_onmt import TransformersONMT

# instantiate the class
transformers_onmt = TransformersONMT(
    path_to_folder='../', 
    dataset_name='USPTO_test', 
    experiment_name='test', 
    src_train_path=src_train_path_1000, 
    tgt_train_path=tgt_train_path_1000, 
    src_val_path=src_val_path_1000, 
    tgt_val_path=tgt_val_path_1000, 
    src_test_path=src_test_path_1000, 
    tgt_test_path=tgt_test_path_1000,  
)
```
And this allows us to entirely preprocess, train, and infer with minimal use of mandatory arguments (optional keyboard arguments are still available)
```
# build vocab
transformers_onmt.preprocess()

# train the model (again with optional argument train_steps set to 1000)
transformers_onmt.train(train_steps=1000)

# Perform inference (here we have to give the path of the model we want to use)
transformers_onmt.translate(path_model='../results/transformer_model/USPTO_test/test/checkpoints/model_test_1000.pt')
```
## Choosing the best model with Tensorboard
To choose the best model for upcoming inference, we can use tensorboard. The files of interest are stored in the `tensorboard` folder. In the example, if you `cd ../results/transformer_model/USPTO_test/test/tensorboard/` and then you enter the command `tensorboard --logdir /path_to_the_folder_of_interest/` you will have each checkpoint displayed in the http://localhost:6006 
accessible from your navigator as long as the terminal is not closed.



