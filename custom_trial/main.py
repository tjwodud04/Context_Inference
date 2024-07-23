from model import load_model
from datasets import Dataset
from data import CustomDataset, DataCollatorForSupervisedDataset
from train import train
from save import merge_and_save

def data_call(tokenizer) :
    train_dataset = CustomDataset("../data/대화맥락추론_train.json", tokenizer)
    valid_dataset = CustomDataset("../data/대화맥락추론_dev.json", tokenizer)

    train_dataset = Dataset.from_dict({
        'input_ids': train_dataset.inp,
        "labels": train_dataset.label,
        })
    valid_dataset = Dataset.from_dict({
        'input_ids': valid_dataset.inp,
        "labels": valid_dataset.label,
        })
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return train_dataset, valid_dataset, data_collator

# def main_function(base_model_path, name, new_merged_model_path):
def main_function(base_model_path, name):
    model, tokenizer, peft_config = load_model(base_model_path)
    train_dataset, valid_dataset, data_collator = data_call(tokenizer)
    train(
        model=model, 
        tokenizer=tokenizer, 
        train_dataset=train_dataset,
        valid_dataset = valid_dataset,
        data_collator = data_collator,
        peft_config=peft_config,
        adapter_output_dir=adapter_output_dir, 
        new_pretrained_model_dir=new_pretrained_model_dir
        )
    # merge_and_save(new_merged_model_path)


if __name__ == "__main__" :
    base_model_path = "./base_model/solar_v01"
    name = "solar_v01"
    adapter_output_dir = f"./results/{name}/Adapter/"
    new_pretrained_model_dir = f"./results/{name}/pretrained/"
    # new_merged_model_path = f"./result/{name}/final/"
    main_function(base_model_path, adapter_output_dir)
    # main_function(base_model_path, adapter_output_dir, new_merged_model_path)