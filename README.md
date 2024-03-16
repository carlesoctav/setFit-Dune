# Few-shot Learning to do sentence classification on movies reviews

## How to install
Create your virtual environment using your preferred method. Once created, activate it and execute the following commands:
```bash
pip install  -r requirement.txt
```

## about Datasets
The datasets are scraped from IMDb reviews. You can actually recreate them with:
```bash
python src/make_dataset.py [ID]
```
where ID is the unique movie ID given by IMDb.

Please note that we use Selenium for automation, and Selenium requires a matching version of Google Chrome and the Google Chrome driver. You can install it using [setup.sh](setup.sh), but please be aware that it may affect your Google Chrome installation (I personally use Firefox as my main browser).

The structure of the datasets is as follows:

```json
{
    id:
    title:
    content:
    rating:
    sentiment:
    date:
    name:
}
```
The sentiment variable is binary. I assume that if someone rates < 5, then it's a negative sentiment; otherwise, it's positive (just some heuristic thinking here).

## Text Processing
```python
def text_cleaning(input_string: str):
    lowercase = input_string.lower()
    remove_accented = (
        unicodedata.normalize("NFKD", lowercase)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    remove_extra_whitespace = re.sub(r"^\s*|\s\s*", " ", remove_accented).strip()
    return remove_extra_whitespace

def process_datasets(batch):
    return {
        "text": text_cleaning(batch["title"] + " " + batch["content"]),
        "label": batch["sentiment"],
```

We concatenate the title and content to form a single text. We perform some cleaning by making it lowercase and removing annoying unicode symbols (such as emoticons if present), as well as removing trailing/extra whitespace. We don't really care about the number of words here because during training, it will be truncated to the `max_seq_length` value, which is typically set to 512.


## Why Few-shot learning
The idea behind SetFit is to train a classifier with decent performance even when trained using a small sample size. We use a pretrained model that has been trained on contrastive loss for sentence similarity tasks and then finetuned on our custom datasets.

TLDR, here's how it works:
1. Create positive and negative sentence pairs, where positive pairs have similar sentiment ([read more about this idea here](https://huggingface.co/docs/setfit/conceptual_guides/sampling_strategies)).
2. Train the model using contrastive loss, hoping to create embedding vectors that are even more separable.
3. Freeze the transformer's body and train the head (classifier). I use logistic regression here.

Read more about the idea [here](https://huggingface.co/docs/setfit/conceptual_guides/setfit). You can also refer to [the original paper](https://arxiv.org/pdf/2209.11055.pdf).

So why use few-shot learning? It achieves competitive results with smaller datasets and can be trained using a CPU. This is in contrast to fine-tuning the entire model, which requires a larger amount of data (around 100-1000 samples). It is also better than training only the head using the vector representation [CLS] token from vanilla transformers (trained on MLM tasks) as an input, as demonstrated in the script [here](src/simple_classifer_with_bert.py) (which only achieves 0.53 accuracy with 8 samples), compared to SetFit's accuracy of 0.8xx.

## How to train
```bash
usage: train.py [-h] --config-file CONFIG_FILE

options:
  -h, --help            show this help message and exit
  --config-file CONFIG_FILE
```
some example of the config file:
```yaml
exp_name: dune2-sent-classfier-8-shot
pretrained_model: paraphrase-mpnet-base-v2
language: en
optimizer: AdamW
lr: 2e-5
epochs: 1
batch_size: 16
num_pairs_contrastive: 10 # This parameter is not used yet
second_phase_end_to_end: False # This parameter is not used yet
train_set: data/raw/15239678.jsonl
val_ratio: 0.2
test_set:
few_shot: 8
max_seq_length: 256
hf_push_to_hub: carlesoctav/SentimentClassifierDune-8shot
```

If `hf_push_to_hub` is present, you need to log in to Hugging Face before that.
```bash
huggingface-cli login
```

## Results
Here are the results that have been trained on the Dune dataset with 8-shot and 64-shot:

| Model | Label | Accuracy | Precision | Recall | F1 |
|-------|-------|----------|-----------|--------|----|
| 8-shot | all | 0.8247 | 0.9916 | 0.8188 | 0.8969 |
| 64-shot | all | 0.8571 | 0.9960 | 0.8512 | 0.9179 |

Achieving accuracies of 0.82 and 0.85 with only 8 and 64 samples, respectively, while trained on 1 epoch (using contrastive loss) and 3000+ iterations of logistic regression, i think it's impressive!

Check the available models here:
- [SentimentClassifierDune64shot](https://huggingface.co/carlesoctav/SentimentClassifierDune64shot)
- [SentimentClassifierDune-8shot](https://huggingface.co/carlesoctav/SentimentClassifierDune-8shot)

```yaml
exp_name: dune2-sent-classifier-8-shot-then-barbie
pretrained_model: carlesoctav/SentimentClassifierDune64shot
language: en
optimizer: AdamW
lr: 2e-5
epochs: 1
batch_size: 16
num_pairs_contrastive: 10
second_phase_end_to_end: False
train_set: data/raw/barbie.jsonl
val_ratio: 0.2
test_set: data/raw/15239678.jsonl
few_shot: 64
max_seq_length: 256 # This parameter is not used yet, so the maximum length remains 512.
hf_push_to_hub: carlesoctav/SentimentClassifierBarbieDune-8shot
```

## Retrained Model Incorporating New Datasets

We can utilize the `train.py` script with an updated configuration file. Below is the YAML configuration:

```yaml
exp_name: dune2-sent-classifier-8-shot-then-barbie
pretrained_model: carlesoctav/SentimentClassifierDune64shot
language: en
optimizer: AdamW
lr: 2e-5
epochs: 1
batch_size: 16
num_pairs_contrastive: 10
second_phase_end_to_end: False
train_set: data/raw/barbie.jsonl
val_ratio: 0.2
test_set: data/raw/15239678.jsonl
few_shot: 64
max_seq_length: 256 # This parameter is not used yet, so the maximum length remains 512.
hf_push_to_hub: carlesoctav/SentimentClassifierBarbieDune-8shot
```

As evident, we utilize `carlesoctav/SentimentClassifierDune64shot` as the pretrained model, which has been trained on the Dune dataset. Additionally, with the `test_set` specified in the configuration, we will evaluate the trained model on this older Dune dataset. The performance metrics are presented in the table below:

| Dataset | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|--------|----------|
| Barbie  | 0.826    | 0.987     | 0.798  | 0.882    |
| Dune    | 0.881    | 0.995     | 0.876  | 0.932    |

## Q and A

1. How did you vectorize the reviews? Justify your choice of algorithm.
    I use contextualized vectors from the output of transformers, especially the [CLS] token. However, we don't use vectors from a model trained on MLM tasks (as explained in previous sections on why I use setfit). Instead, we retrain the representation using contrastive loss. Why not a bag of words (BoW) or other sparse representations? From what I've learned, BoW requires a large amount of text for the statistics to work well. Utilizing a pretrained model that is readily available from Hugging Face is a more cost-effective and easier approach (thanks to the wonderful community).
2. How would you calculate the accuracy of your sentiment predictor?
    Since the output of the logistic head is a value between 0 and 1, we assume any value greater than 0.5 represents a positive sentiment (1), otherwise it represents a negative sentiment. I can calculate the accuracy as (TN + TP) / num_of_samples, where TN is the number of true negatives and TP is the number of true positives.
3. How would you calculate the speed of inference?
    To calculate the speed of inference, you can run the inference multiple times and capture the mean and standard deviation. You can refer to the [benchmark.py](src/benchmark.py) script for an example.
4. What would be your next steps to improve the solution?
    One possible next step to improve the solution is to enhance interpretability. This can be achieved by using techniques such as integrated gradients or finding tokens that have higher similarity to the text representation, indicating their importance.
