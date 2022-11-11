from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from datasets import Dataset

def tokenize(dataset) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 

    def tokenize_function(sample):
        return tokenizer(sample["sentence"], max_length=512, padding="max_length", truncation=True)

    print("Tokenizing...")
    tokenized_dataset = dataset.map(tokenize_function)
    tokenized_dataset = tokenized_dataset.remove_columns(["sentence"])
    tokenized_dataset["train"] = tokenized_dataset["train"].rename_column("label","labels")
    tokenized_dataset["dev"] = tokenized_dataset["dev"].rename_column("label","labels")
    tokenized_dataset.set_format("torch")

    return tokenized_dataset

if __name__=="__main__":
    from datasets import load_dataset

    dataset = load_dataset("wikipedia", "20220301.en")

    from transformers import AutoTokenizer
    import os

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") 

    def tokenize_function(sample):
        return tokenizer(sample["text"])

    print("Tokenizing...")
    for i in range(0,len(dataset["train"]),1000000):
        print(i)
        tokenized_dataset = dataset["train"].select(range(i,i+1000000)).map(tokenize_function, num_proc=16)
        tokenized_dataset.save_to_disk(f"../wikipedia_dataset/tokenized{i}")
        os.system("rm /home/fwe21/.cache/huggingface/datasets/wikipedia/20220301.en/2.0.0/aa542ed919df55cc5d3347f42dd4521d05ca68751f50dbc32bae2a7f1e167559/tmp*")
        os.system("rm /home/fwe21/.cache/huggingface/datasets/wikipedia/20220301.en/2.0.0/aa542ed919df55cc5d3347f42dd4521d05ca68751f50dbc32bae2a7f1e167559/cache*")
