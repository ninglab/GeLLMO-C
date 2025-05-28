import os
import sys
import time
import pandas as pd
import json


import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from datasets import load_dataset, Dataset

from prompter import Prompter
from config import DEFAULT_MAX_NEW_TOKENS, TARGET_TASKS
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

## Load merged model
    
def main(
    load_in_8bit: bool = False,
    use_lora: bool = True,
    base_model: str = "",
    lora_weights: str = "",
    opt_type: str = "thresh",
    task: str = "",
    setting: str = "seen",
    data_path: str = "",
    output_dir: str = "",
    #test_sample_is_pair: bool = False,
    batch_size: int = 16,
    num_return_sequences: int = 1,
):
    # base_model = base_model or os.environ.get("BASE_MODEL", "")
    response_split = "%%% Response:"
    prompter = Prompter(opt_type, response_split)

    #test_data = load_dataset('json', data_files=f'{data_path}', split='train')
    #test_data = pd.DataFrame(test_data)
    task = TARGET_TASKS[task]
    #print(task)
    test_data = load_dataset(data_path)['test']
    #print(len(test_data))
    test_data = test_data.filter(lambda example: example['property_comb'] == task and example['instr_setting'] == setting)
    #if "+" not in task:
    #test_data = test_data[test_data['instr_setting'] == setting]
    #else:
    #    test_data = test_data[(test_data['task'] == task) & (test_data['instr_setting'] == setting)]
        
    #test_data = Dataset.from_pandas(test_data)
    print(task, setting, len(test_data))

    if base_model == 'Llama-3.1-8B-Instruct' or base_model == 'Llama-3.1-70B-Instruct':
        base_model = f'meta-llama/{base_model}'
    elif base_model == 'Mistral-7B-Instruct-v0.3':
        base_model = f'mistralai/{base_model}'

    if "Llama-3.1-70B" in base_model:
        print("Using 70b model, loading in 8bit")
        load_in_8bit = True
        batch_size = 4

    instructions = []
    for data_point in test_data:
        full_prompt = prompter.generate_prompt(
            data_point,
            prompt_type = 'ins',
            add_response=False
        )
        instructions.append(full_prompt)

    print(instructions[0])

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    quantization_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )

    if use_lora:
        model = PeftModel.from_pretrained(model,
                                          lora_weights, 
                                          torch_dtype=torch.bfloat16,
                                          device_map="auto",)

    if not load_in_8bit:
        model.bfloat16()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if not model.config.eos_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'
        model.config.eos_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = model.config.eos_token_id
        tokenizer.padding_side = 'left'

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.float16, 
        device_map="auto",
    )
    # without below, batch size > 1 will throw error
    # this is a hack to make it work for Llama models -- some issue with transformers version
    if "Llama-3.1-8B" in base_model:
        print("Setting pad token id to eos token id for batching")
        pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id[0]

    results = []
    max_batch_size = batch_size
    for i in range(0, len(instructions), max_batch_size):
        instruction_batch = instructions[i:i + max_batch_size]
        #print(f"Processing batch {i // max_batch_size + 1} of {len(instructions) // max_batch_size + 1}...")
        start_time = time.time()
    
        batch_results = evaluate(prompter, instruction_batch, tokenizer, pipe, max_batch_size, 
                                num_return_sequences)
            
        results.extend(batch_results)
        print(f"Finished processing batch {i // max_batch_size + 1}. Time taken: {time.time() - start_time:.2f} seconds")

    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{task}_response.json", 'w') as f:
        json.dump(results, f)


def evaluate(prompter, 
             prompts, tokenizer, pipe, batch_size, 
             num_return_sequences=1,
             max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
             do_sample=False,
             temperature=1,
            ):
    batch_outputs = []

    generation_output = pipe(
        prompts,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        #top_p=0.95,
        num_return_sequences=num_return_sequences,
        num_beams=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id,
        batch_size=batch_size,
    )

    for i in range(len(generation_output)):  
        #print(prompts[i])
        #print(generation_output[i][0])
        responses = []
        for j in range(num_return_sequences):
            responses.append(prompter.get_response(generation_output[i][j]['generated_text']))
            #resp = prompter.get_response(generation_output[i][0]['generated_text'])
        output = {'prompt': prompts[i], 'response': responses}
        batch_outputs.append(output)

    return batch_outputs


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)