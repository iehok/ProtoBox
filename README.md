# ProtoBox

## Before Training

Before training, please do the following. 
1. Set specifying paths
2. Download WSD datasets
3. Prepare the training data, development data, and test data

### 1. Set specifying paths
You can set some important paths by editting the `code/config.py`. 
`ROOT_DIR` is the root directory in your environment. 
`EXP_DIR` is the directory where trained models are saved. 
`DATA_DIR` is the directory where datasets are stored. 

### 2. Download WSD datasets
Following previous WSD researches, we use this [WSD Framework](http://lcl.uniroma1.it/wsdeval/). 
So, please download the datasets. 
After that, please move the downloaded datasets to `DATA_DIR`. 

### 3. Prepare the training data, development data, and test data
We train the model with the instances that contain the hyponyms of the root sense. 
And we evaluate trained models by three tasks: Word Sense Disambiguation, New Sense Classification, and Hypernym Identification. 
So, please execute following command to prepare these datasets. 
```
python -m code.prepare_data --root animal.n.01
```
You can change the root sense by editting the `--root` option. 

## Run Training
You can run the training process by the following command. 
```
bash scripts/train.sh
```

## Run Evaluation
You can run the evaluation process by the following command. 
You change the task by editting the `EXPERIMENT` value in the `scripts/test.sh`. 
```
bash scripts/test.sh
```

## Contact
If you have any questions or comments, please email Kohei Oda (s2420017@jaist.ac.jp). 
