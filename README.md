# English to Telugu Translation Project

## Overview
This project focuses on translation from English to Telugu using the machine translation models. We have fine-tuned the following models on the Samanantar dataset:
- MBART50
- MT5
- NLLB

We selected a subset of 60,000 parallel corpora from the available 5 million parallel corpora for our training data  and FLORES22 as evaluation set.

## Models
### [MBART50](https://arxiv.org/pdf/2008.00401.pdf)

#### Model Overview:
[mBART50](https://arxiv.org/pdf/2008.00401.pdf) (Multilingual BART) is a multilingual extension of the BART (Bidirectional and Auto-Regressive Transformers) model. It is specifically designed for multilingual sequence-to-sequence tasks and demonstrates strong performance across various languages. By pretraining on multiple languages simultaneously, mBART learns a shared cross-lingual representation space, enabling effective transfer learning across languages.

#### Fine-Tuning Process:
For [fine-tuning mBART](https://huggingface.co/transformers/v4.7.0/model_doc/mbart.html) on our task, we utilized the Samanantar dataset. This dataset is rich in multilingual text data, making it suitable for training multilingual models like mBART. The fine-tuning process involves adapting the pretrained mBART model to our specific task by updating its parameters based on the Samanantar dataset. We have fine-tuned the model for only one epoch and were able to achieve a SacreBLEU score of approximately 9 using a batch size of 8, a learning rate of 5e-4, and the Adam optimizer. We set MAX_INPUT_LENGTH and MAX_TARGET_LENGTH to 128. The training stage took around 35 minutes for one epoch on one NVIDIA A100 GPU. We observed that training for more epochs could increase the quality of translation. The code is present here [link](./MBart50model.py)


### [MT5](https://arxiv.org/abs/2010.11934)

#### Model Overview:
MT5 (Multilingual T5) is a multilingual variant of the T5 (Text-To-Text Transfer Transformer) model, which is a powerful text-to-text transformer architecture. MT5 is designed to handle multilingual tasks, allowing for seamless processing of text in multiple languages within a single model. By pretraining on a diverse set of languages, MT5 learns to understand and generate text in various languages, enabling effective transfer learning across linguistic boundaries.

#### Fine-Tuning Process:
To [fine-tune MT5](https://huggingface.co/docs/transformers/v4.14.1/en/model_doc/mt5) for our specific task, we followed a similar approach to the pretrained mBART model. We utilized a multilingual dataset suitable for our task and fine-tuned MT5 on this dataset. The fine-tuning process involves updating the parameters of the pretrained MT5 model based on the task-specific dataset, enabling the model to adapt its knowledge to our target task.  The code is present here [link](./mT5model.py)


### [NLLB](https://arxiv.org/abs/2207.04672)
- For details about NLLB see - https://huggingface.co/docs/transformers/en/model_doc/nllb
- We [fine-tuned](https://discuss.huggingface.co/t/fine-tuning-nllb-model/31237) NLLB model after preprocessing unknown tokens in Telugu language on Samanantar dataset with configurations as follows:
1)Train split= 0.8 , test split = 0.2 on subset of Samanatar dataset 
2)used Adam with initial lr= 2e-4 and batchsize of 8.
3)epoch =1.

  The code is present here [link](./NLLB_finetunedmodel.ipynb)

### [Llama-7B](https://arxiv.org/abs/2307.09288)

 

## Evaluation Metrics
Our primary evaluation metric is the [SacreBLEU](https://github.com/mjpost/sacrebleu) score.

## Data
A data sample of 500 parallel corpus of English and Telugu sentences  taken from [Samanantar](https://ai4bharat.iitm.ac.in/samanantar/) dataset  is available in the [`data`](./Data).


## Conclusion and Future Work
As of now  we have exploited the transformer models like mBART50 , NLLB , mT5  and
have fine tuned the models using 60000 training samples taken from samanantar dataset
and was able to acheive decent BLEU score.For the training we trained the model with 1
epoch.

For future work, we plan to delve deeper into the exploration of additional pre-trained
large language models (LLMs), such as LLaMA2, coupled with [LoRA fine-tuning](https://arxiv.org/abs/2106.09685)
techniques. This approach aims to further enhance the performance of our translation
models by exploiting the strengths of diverse pre-trained architectures.
Additionally, we intend to experiment with training our models using custom tokenization
strategies specially designed to the characteristics of the English to Telugu translation
task. By adapting tokenization of the target language, we anticipate improvements in
translation accuracy and efficiency. Furthermore, we aim to augment our training dataset
by increasing the number of samples, thereby providing the model with a richer and more
diverse set of contexts to learn from. We will try to train the model with more no of epochs
for increasing the performance of the model. We will also consider different metrics like
ChrF++ ,ROUGE score  for evaluation the quality of translation



## Acknowledgements
This project is contributed to by the following collaborators:

- [Alokam Gnaneswara Sai](https://github.com/alokamgnaneswarasai)
- [Onteru Prabhas Reddy](https://github.com/prabhas2002)
- [Pallekonda Naveen Kumar](https://github.com/PNaveenKumar1)


## References 

Note: We have made an effort to refer to as many sources as possible for this document. We apologize for any oversights.



