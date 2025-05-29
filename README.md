# GeLLMO-C: Large Language Models for Controllable Multi-property Multi-objective Molecule Optimization
We introduce C-MuMOInstruct, the first instruction-tuning dataset focused on multi-property optimization with explicit, property-specific objectives. Leveraging C-MuMOInstruct, we develop GeLLM4O-Cs, a series of instruction-tuned LLMs that can perform targeted property-specific optimization. Our experiments across 5 in-distribution and 5 out-of-distribution tasks show that GeLLM4O-Cs consistently outperform strong baselines, achieving up to 126% higher success rate. Notably, GeLLM4O-Cs exhibit impressive 0-shot generalization to novel optimization tasks and unseen instructions.

## Requirements

Please use `pip install -r requirements.txt` to install dependencies. Ensure you have `python >= 3.10.0` installed.


## Dataset

The dataset with all training and test JSON files is available in [Huggingface](https://huggingface.co/datasets/NingLab/C-MuMOInstruct). 

## Models

The instruction-tuned model checkpoints will be uploaded to HuggingFace soon. 


## Training

To instruction-tune the base models, run the following:
```
bash train.sh $base_model NingLab/C-MuMOInstruct $expt_dir $num_epochs $max_steps $tasks $batch_size
```
- `$base_model` specifies the base model name as in Huggingface (either `Mistral-7B-Instruct-v0.3` or `Llama-3.1-8B-Instruct`)
- `$data_dir` specifies the path to the local data directory
- `$expt_dir` specifies the path to the directory where the finetuned checkpoints and lora weights will be saved
- `$num_epochs` specifies the number of epochs
- `$max_steps` specifies the number of gradient update steps. By default, this will override `$num_epochs` argument. Set this to 0 if you don't want it to override `$num_epochs` 
- `$tasks` specifies the list of tasks to be used for instruction-tuning
    - example 1: "['bbbp+plogp+qed']" to train a specialist model optimizing only this property combination
    - example 2: "['bbbp','plogp','qed','bbbp+qed','bbbp+plogp','plogp+qed','bbbp+plogp+qed']" to train a generalist model on the superset of 3 properties: BBBP, DRD2 and QED
    - example 3: "[]" to train the generalist model GeLLMO-C-P(10) which uses all possible property combinations of up to 10 properties excluding the combinations in OOD tasks. The exclusion logic is already incorporated within `trainer.py`
- `$batch_size` specifies the batch size. We use 32 for specialist models and 128 for generalist ones


## Inference

For model inference, run the following command:

```
bash inference.sh $base_model NingLab/C-MuMOInstruct $lora_weights $output_dir $task $num_seq $setting
```
- `$base_model` specifies the base model (either `Mistral-7B-Instruct-v0.3` or `Llama-3.1-8B-Instruct`)
- `$data_dir` specifies the path to the local data directory
- `$lora_weights` specifies the local path or the HuggingFace model hub containing the adapter weights of the instruction-tuned model
- `$output_dir` specifies the output path where the LLM generated responses will be stored in JSON format
- `$task` specifies the 3/4/5-lettered task ID as listed in the paper (i.e., BPQ, BDPQ, etc..)
- `$num_seq` specifies the number of generated responses for each prompt (we set it to 20 in our experiments)
- `$setting` specifies whether to use seen or unseen instruction (use `unseen` to evaluate on the unseen instruction setting)

Example: to run a Mistral-7B tuned LLM on the task BDPQ under the seen instruction setting:
```
bash inference.sh Mistral-7B-Instruct-v0.3 NingLab/C-MuMOInstruct NingLab/GeLLMO-P10-Mistral /tmp/ BDPQ 20 seen
```