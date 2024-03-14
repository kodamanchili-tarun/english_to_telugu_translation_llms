# English to Telugu Translation Project

## Overview
This project focuses on translation from English to Telugu using the machine translation models. We have fine-tuned the following models on the Samanantar dataset:
- MBART50
- MT5
- NLLB

We selected a subset of 60,000 parallel corpora from the available 5 million parallel corpora for our training data  and FLORES22 as evaluation set.

## Models
### MBART50

#### Model Overview:
mBART (Multilingual BART) is a multilingual extension of the BART (Bidirectional and Auto-Regressive Transformers) model. It is specifically designed for multilingual sequence-to-sequence tasks and demonstrates strong performance across various languages. By pretraining on multiple languages simultaneously, mBART learns a shared cross-lingual representation space, enabling effective transfer learning across languages.

#### Fine-Tuning Process:
For fine-tuning mBART on our task, we utilized the Samanantar dataset. This dataset is rich in multilingual text data, making it suitable for training multilingual models like mBART. The fine-tuning process involves adapting the pretrained mBART model to our specific task by updating its parameters based on the Samanantar dataset.


### MT5
- Details about fine-tuning process
- Any specific configurations

### NLLB
- We fine-tuned this model after preprocessing unknown tokens in Telugu language on Samanantar dataset with configurations as follows:
1)Train split= 0.8 , test split = 0.2 on subset of Samanatar dataset 
2)used Adam with initial lr= 2e-4 and batchsize of 8.
3)epoch =1.
  

## Evaluation Metrics
Our primary evaluation metric is the BLEU score.

## Data
A data sample of 500 parallel sentences is available in the [`data`](./Data).


## Acknowledgements
This project is contributed to by the following collaborators:

- [Alokam Gnaneswara Sai](https://github.com/alokamgnaneswarasai)
- [Onteru Prabhas Reddy](https://github.com/prabhas2002)
- [Pallekonda Naveen Kumar](https://github.com/PNaveenKumar1)






