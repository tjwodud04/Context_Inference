import argparse
import json
import tqdm

import torch
import numpy
from transformers import AutoTokenizer, AutoModelForCausalLM

from data import CustomDataset


# fmt: off
# parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

# g = parser.add_argument_group("Common Parameter")
# g.add_argument("--output", default="/home/work/jysuh/Trial/Third_trial/results/Bllossom/test_result/result.json" ,type=str, required=True, help="output filename")
# g.add_argument("--model_id", default="/home/work/jysuh/Trial/Third_trial/results/Bllossom/final/", type=str, required=True, help="huggingface model id")
# g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
# g.add_argument("--device", default="cuda:0", type=str, required=True, help="device to load the model")
# fmt: on

output_file = "/home/work/jysuh/Trial/Third_trial/results/Bllossom/test_result/result.json"
model_dir = "/home/work/jysuh/Trial/Third_trial/results/Bllossom/final/"
device = "cuda:0"


def main(output_file, model_dir, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()


    # if tokenizer == None:
    tokenizer = model_dir

    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = CustomDataset("/home/work/jysuh/Trial/data/대화맥락추론_test.json", tokenizer)

    answer_dict = {
        0: "inference_1",
        1: "inference_2",
        2: "inference_3",
    }

    with open("/home/work/jysuh/Trial/data/대화맥락추론_test.json", "r") as f:
        result = json.load(f)

    for idx in tqdm.tqdm(range(len(dataset))):
        inp, _ = dataset[idx]
        outputs = model(
            inp.to(device).unsqueeze(0)
        )
        logits = outputs.logits[:,-1].flatten()
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(
                    [
                        logits[tokenizer.vocab['A']],
                        logits[tokenizer.vocab['B']],
                        logits[tokenizer.vocab['C']],
                    ]
                ),
                dim=0,
            )
            .detach()
            .cpu()
            .to(torch.float32)
            .numpy()
        )

        result[idx]["output"] = answer_dict[numpy.argmax(probs)]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    # exit(main(parser.parse_args()))
    output_file = "/home/work/jysuh/Trial/Third_trial/results/Bllossom/test_result/result.json"
    model_dir = "/home/work/jysuh/Trial/Third_trial/results/Bllossom/final/"
    device = "cuda:0"
    main(
        output_file=output_file,
        model_dir=model_dir,
        device=device
        )