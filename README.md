# English to Telugu Translation Project

## Overview
This project focuses on translation from English to Telugu using the machine translation models. We have fine-tuned the following models on the Samanantar dataset:
- MBART50
- MT5
- NLLB

We selected a subset of 60,000 parallel corpora from the available 5 million parallel corpora for our training data  and FLORES22 as evaluation set.

## Models
### MBART50
- Details about fine-tuning process
- Any specific configurations

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






