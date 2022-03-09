# IT5: Large-scale Text-to-text Pretraining for Italian Language Understanding and Generation 🇮🇹

[Gabriele Sarti](https://gsarti.com) • [Malvina Nissim](https://malvinanissim.github.io/)

> **Abstract:** The T5 model and its unified text-to-text paradigm contributed in advancing the state-of-the-art for many natural language processing tasks. While some multilingual variants of the T5 model have recently been introduced, their performances were found to provide suboptimal performances for languages other than English if compared to ad-hoc monolingual variants. Motivated by these findings we introduce IT5, the first family of encoder-decoder transformer models pretrained specifically on the Italian language. We perform a thorough cleaning of a web-crawled Italian corpus including more than 40 billion words, and use it to pretrain three IT5 models of different sizes. We then evaluate the performance of the IT5 models and their multilingual counterparts on a broad range on natural language understanding and generation benchmarks for Italian. We find the monolingual IT5 models to provide the best scale-to-performance ratio across tested models, consistently outperforming their multilingual counterparts and setting a new state-of-the-art for most Italian conditional language generation tasks.

This repository groups links and materials for the paper ["IT5: Large-scale Text-to-text Pretraining for Italian Language Understanding and Generation"](https://arxiv.org/abs/2203.03759). If you use any of the following contents for your work, you are kindly requested to cite our paper:

```bibtex
@article{sarti-nissim-2022-it5,
    title={{IT5}: Large-scale Text-to-text Pretraining for Italian Language Understanding and Generation},
    author={Sarti, Gabriele and Nissim, Malvina},
    journal={ArXiv preprint 2203.03759},
    url={https://arxiv.org/abs/2203.03759},
    year={2022},
	month={mar}
}
```

## Pre-training Materials

- The repository [gsarti/t5-flax-gcp](https://github.com/gsarti/t5-flax-gcp) provides the script and a detailed explanation of the pre-training process using Huggingface and Flax on a TPU v3-8 VM via Google Cloud Platform.

- The Cleaned Italian mC4 Corpus used for pre-training the IT5 models is made available on the Huggingface Datasets Hub under the identifier [gsarti/clean_mc4_it](https://huggingface.co/datasets/gsarti/clean_mc4_it).

- The following pre-trained IT5 models are made available vie the Huggingface Models Hub:

    - [IT5 Small](https://huggingface.co/gsarti/it5-small), encoder-decoder with 6+6 layer and 60M parameters.

    - [IT5 Base](https://huggingface.co/gsarti/it5-base), encoder-decoder with 12+12 layer and 220M parameters.

    - [IT5 Large](https://huggingface.co/gsarti/it5-large), encoder-decoder with 24+24 layer and 738M parameters.

## Experiments Materials

It is not possible for us to freely release the fine-tuning data due to access restrictions imposed by some of the original dataset creators. Please reach out at [gabriele.sarti996@gmail.com](mailto:gabriele.sarti996@gmail.com) showing proof of having received access to the XFORMAL dataset ([procedure here](https://github.com/Elbria/xformal-FoST)) and we will be happy to provide with the preprocessed data.

This repository contains the following materials to reproduce fine-tuning experiments and evaluation:

- The folder [finetuning](finetuning) contains the [run_seq2seq.py](finetuning/run_seq2seq.py) used to fine-tune the models on the different tasks and multiple helper files used to parametrize and run the experiments in a SLURM cluster.

- The folder [inference](inference) contains the [infer.py](inference/infer.py) used to predict the outputs of all tested models on all datasets and multiple helper files used to parametrize and run inference in a SLURM cluster.

- The folder [model_predictions](model_predictions) contains all the predictions produced with the inference script for all models and tested datasets in text one-line-per-example format.

- The notebook [compute_scores.ipynb](compute_scores.ipynb) contains the code used to evaluate the performances of all the models on all the datasets. The configuration [bertscore_baseline_ita.tsv](bertscore_baseline_ita.tsv) is used in the notebook to compute the renormalized BERTScore values.

We release all the **45 fine-tuned model checkpoints** (3 IT5 models and 2 mT5 models on a total of 9 tasks) in the [it5 repository](https://huggingface.co/it5) on Huggingface. All models include Tensorboard logs for the fine-tuning procedure and are available for usage with the Huggingface Transformers library using Tensorflow, Pytorch and JAX. They can be used directly with `pipelines` as:

```python
from transformers import pipelines

# e.g. to load IT5 Small trained on formal-to-informal style 
# transfer, use `it5/it5-small-formal-to-informal`
f2i = pipeline("text2text-generation", model='it5/it5-small-formal-to-informal')
f2i("Vi ringrazio infinitamente per vostra disponibilità")
>>> [{"generated_text": "e grazie per la vostra disponibilità!"}]
```

or loaded separately as:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# e.g. to load IT5 Small trained on headline generation,
# use `it5/it5-small-headline-generation` as MODEL ID.
tokenizer = AutoTokenizer.from_pretrained("<MODEL ID>")

model = AutoModelForSeq2SeqLM.from_pretrained("<MODEL ID>")
```

Refer to the individual model cards on the Model Hub and the original paper for more details.

