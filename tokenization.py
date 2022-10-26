from transformers import AutoTokenizer

def tokenize(dataset):
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