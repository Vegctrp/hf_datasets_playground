"""very easy sample program of hf_datasets."""

from datasets import load_dataset

# from huggingface_hub import list_datasets  # noqa: ERA001
from transformers import AutoTokenizer

from hf_datasets_playground.utils.logger import get_logger

logger = get_logger(name=__name__, level=10)

# logger.info([dataset.id for dataset in list_datasets()])  # noqa: ERA001
squad_dataset = load_dataset("squad")

logger.info(squad_dataset)
# logger.info(squad_dataset)  # noqa: ERA001
logger.info(squad_dataset["train"][0])

dataset_with_length = squad_dataset.map(lambda x: {"length": len(x["context"])})
logger.info(dataset_with_length)

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokenized_dataset = squad_dataset.map(lambda x: tokenizer(x["context"]), batched=True)
logger.info(tokenized_dataset)
