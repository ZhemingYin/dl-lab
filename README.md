# Team06
- Zheming Yin (st178328)
- Ziheng Tong (st174258)

# How to run the code
## Diabetic Retinopathy Detection
### Default setting:

The code is set to train the binary classification of the IDRID dataset on the server as default. The VGG model is
used in this case. 


### Training
- Set the macro parameter `CLASSIFICATION` in the `config.gin` to `"binary"` or `'multiple'` for 2-class or 5-class 
classification, and then type in the directory that stored the dataset. The other parameters like learning rate, 
batch size, etc., can also be modified directly in the configuration file. You can also choose the dataset that you want
to run, either `'IDRID'` or `'eyepacs'`. 

- Make sure the FLAG for `'train'` in the `main.py` is `True`, then select the model you want to use and specify
how many epochs you want to run. You can then start the training process by running the
`main.py` directly. 

### Evaluation
Keep the configured parameters the same as in the training process. You just need to set
the FLAG for `'train'` in the `main.py` to `False`. Also remember to change the path where stored the checkpoint in the
`evaluation.py`. 

### Hyperparameter tuning
We also provide the possibility of hyperparameter tuning using W&B in our project. 
In order to tune the hyperparameters of a specific model, just modify the corresponding 
configurations in the `config_wandb.gin` as well as the sweep_config in the `wandb_sweep.py`. 
They are generally the same as the procedure in the training process. Then you can run 
the `wandb_sweep.py` to tune the hyperparameters directly. 

### Ensemble learning
The file `config_ensemble.py` is the configure for ensemble learning, you can modify the type of classification in this file. To run ensemble learning, directly run the file `ensemble_learning.py`.

## Human Activity Recognition
### Default setting
The code is set to train the HAPT dataset on the server using RNN as default. 


### Training
The training process is generally the same as the process in the first project. You can choose whether run the "HAPT" or
"HAR" dataset.
- In this project, you have the global control of several macro parameters such as label type ("s2s" or "s2l"), window 
size and the shift ratio of the windows. 
- For the "HAR" dataset, you have the extra control of training for a single position or combine all the sensors together. 

### Evaluation
The evaluation process is the same as the first project. 

### Hyperparameter optimization
You can optimize the hyperparameters the same as the first project. 

# Results
## Diabetic Retinopathy Detection
### Binary classification
Model|Accuracy|Precision|Recall|F1-score|Sensitivity|Specificity
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Simple CNN|58.3%|54.7%|54.7%|54.7%|71.4%|46.3%
VGG|70.9%|93.8%|93.8%|93.8%|69.8%|76.5%
ResNet101|66.0%|84.4%|84.4%|84.4%|68.4%|58.3%
Inception-Resnet-V2|77.7%|81.2%|81.2%|81.2%|82.5%|70.0%
EnsembleLearning|68.3%|63.1%|63.1%|63.1%|72.3%|46.4%

### 5-class classification
Model|Accuracy|Precision|Recall|F1-score
:---:|:---:|:---:|:---:|:---:
Simple CNN|31.1%|19.1%|28.7%|21.8%
VGG|48.5%|53.2%|38.0%|38.9%
ResNet101|30.1%|24.8%|24.4%|24.0%
Inception-Resnet-V2|43.7%|35.2%|31.9%|31.9%
Ensemble Learning|37.2%|23.9%|26.8%|24.7%

### Hyperparameter optimization using VGG model
Type|Accuracy|Precision|Recall|F1-score|Sensitivity|Specificity
:---:|:---:|:---:|:---:|:---:|:---:|:---:
Binary classification (Tuned)**|**81.6%**|**75.0%**|**75.0%**|**75.0%**|**94.1%**|**69.2%**
5-class classification (Tuned)|53.4%|39.4%|39.9%|39.2% | - | - 

### 5-class classificationEyePACS
Model|Accuracy|Precision|Recall|F1-score
:---:|:---:|:---:|:---:|:---:
VGG|81.5%|51.5%|40.6%|42.8%

## Human Activity Recognition
### HAPT dataset
Model|S2S|S2L
:---:|:---:|:---:
**BRNN**|**95.3%**|**93.6%** 
RNN|93.4%|93.5%
GRU|83.8%|80.7%

### HAR dataset
Position|Test accuracy|F_measure|Recall|Precision
:---:|:---:|:---:|:---:|:---:
**waist**|**84.0%**|**85.7%**|**85.5%**|**87.6%**
shin|80.8%|82.9%|83.2%|83.2%
chest|79.0%|80.6%|81.4%|81.5%
upperarm|76.2%|77.4%|78.8%|77.1%
forearm|69.9%|71.6%|72.2%|73.1%
head|67.9%|70.4%|69.9%|72.0%
thigh|61.1%|63.2%|61.7%|68.7%
multiple|73.0%|70.8%|72.2%|70.8%

