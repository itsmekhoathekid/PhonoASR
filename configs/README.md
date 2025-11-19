# 📖 Configuration Documentation

This section describes the key configuration parameters used for training, inference, and model setup. We use YAML file to control config of models.

## Training Settings

### **Epoch Control**
- `epoch` determines how training is terminated:
  - `epoch = 0`: Training continues until **early stopping** is triggered.
  - `epoch > 0`: Training will run for the specified number of epochs, if early stopping is triggered, it'll stop.
- `patience`: Param controls early stopping (If not specified, the default value is applied).

### **Training Objective**
- `type_training` Specifies the loss function. Supported values:
  - `ce`: Cross-Entropy Loss  
  - `ctc-kldiv`: Combined CTC Loss + KL-Divergence  
  - `transducer`: Transducer Loss, you can also use this to train transducer base models (eg: RNNT, Conv-RNNT, Conformer, ...). When using this, you have to change `type` in `dec` to transducer. 
- If `type_training = ctc-kldiv`, you must include:
  - `ctc-weight`: weighting factor for the CTC component.

### **Training Level**
- `type` defines the linguistic unit used by the model:
  - `word`: Word-level training, It's also applied to our method in phoneme-level cause we use a stack of 3 phonemes units to illustrating a word   
  - `character`: Character-level training  
  - `phoneme`: Phoneme-level training, this phoneme is kinda like character, we break down the word into phoneme unit and separate each word with \<space> token

## Inference Settings

### **Inference Flow**
- `infer` specifies the inference mechanism:
  - `normal`: Standard greedy inference  
  - `mtp_stack`: Multi token prediction inference, this can only be used when traning by our method  

## Model Settings

### **Model Name**
- `model_name`: Defines the name displayed when saving checkpoints.

### **Encoder Configuration**
- Under `enc`:
  - `type`: Specifies the encoder architecture to use.

### **Decoder Configuration**
- Under `dec`:
  - `type`: Specifies the decoder architecture to use. Currently, we have: `base`, `saa_dec`, `vgg_dec`, `transducer`
  - `k`: This param is exclusively used for our method, default is 1, for truly mean phoneme-level as our theory, set it to 3 
