import json
from typing import List
import random
import pandas as pd
import numpy as np

from datasets import load_dataset, Dataset
from config import PROPERTY_FULL_NAMES, PROPERTY_IMPV_THRESHOLDS, DEFAULT_PROPERTY_TEST_THRESHOLDS_UPPER

class Prompter:

    def __init__(self, opt_type: str = 'thresh', 
                 response_split: str = "[/INST] %%% Response:",
                 template_path: str = 'templates/'
                 ):
        # read the prompt and property threshold from json files
        with open(f"{template_path}/{opt_type}.json", "r") as f:
            template_data = json.load(f)

        #self.template_icl = template_data["template_icl"]
        #self.template_zero = template_data["template"]
        self.opt_type = opt_type
        self.system_prompt = template_data["system_prompt"]
        self.task_prompt = template_data["task_prompt"]
        self.task_prompt_llasmol = template_data["task_prompt_llasmol"]
        self.task_prompt_noexplain = template_data["task_prompt_no_explain"]
        self.instructions = template_data["instructions"]
        self.instructions.append(template_data["base_inst"])
        self.adjust_prompt = template_data["adjust_prompt"]

        #self.property_change = template_data["property_change"]
        self.property_change = PROPERTY_IMPV_THRESHOLDS
        
        self.response_split = response_split


    # Function to generate a dynamic instruction with percentage improvements
    def generate_instruction(self, record):
        improved = record["improved"] if 'improved' in record else [prop for prop in record["properties"].keys() if record["properties"][prop]]
        stable = record["stable"] if 'stable' in record else None
        instr_setting = record["instr_setting"]

        # Collect improvement instructions
        improvement_clauses = []
        for prop in improved:
            #print(prop, record["properties"][prop])
            change = "increase" if record["properties"][prop]["change"] > 0 else "decrease"
            direction = "at least" if record["properties"][prop]["change"] > 0 else "at most"
            try:
                value = record["properties"][prop]["target"].__round__(2)
            except:
                value = DEFAULT_PROPERTY_TEST_THRESHOLDS_UPPER[prop]
            # randomly sample a adjust prompt template
            adjust_prompt = random.choice(self.adjust_prompt)
            #percentage = PROPERTY_PERCENT_BUCKETS[level]  # Retrieve predefined threshold
            property = PROPERTY_FULL_NAMES[prop][-1] if instr_setting == 'unseen' else PROPERTY_FULL_NAMES[prop][0]
            # replace the {} in adjust prompt
            if self.opt_type == 'thresh':
                adjust_prompt = adjust_prompt.format(property=property, change=change, direction=direction, value=value)
            elif self.opt_type == 'simple':
                adjust_prompt = adjust_prompt.format(property=property, change=change)
            improvement_clauses.append(adjust_prompt)
        
        # Join improvement clauses with comma separation and an 'and' before the last clause
        if not improvement_clauses:
            improvements_str = ""
        elif len(improvement_clauses) == 1:
            improvements_str = improvement_clauses[0]
        else:
            improvements_str = ", ".join(improvement_clauses[:-1]) + " and " + improvement_clauses[-1]

        # Collect stability constraints
        stable_clause = ""
        if stable and self.opt_type == 'thresh':
            stable_props_names = [PROPERTY_FULL_NAMES[prop][-1] if instr_setting == 'unseen' else PROPERTY_FULL_NAMES[prop][0] for prop in stable]
            stable_props_names = ", ".join(stable_props_names)
            stable_clause = f" while keeping {stable_props_names} unchanged."

        # Construct final instruction
        instruction = improvements_str + stable_clause
        return instruction


    def generate_prompt(self,
                         sample: dict,
                         prompt_type: str,
                         add_response: bool
                         ) -> str:
        """
        Generates a prompt for the given pair of smiles and list of properties with or without a response
        Args:
            sample: A dictionary containing the input smiles, target smiles, and property values
            properties: A list of properties to be changed
            prompt_type: The type of prompt to be generated
            opt_type: The type of optimization (simple or threshold)
            add_response: Whether to add the target smiles to the prompt
        """
        prompt = ""
        input_smiles = sample['source_smiles']
        output_smiles = sample['target_smiles'] if add_response else None
        if prompt_type == 'pt0':
            opt_prompt = self.generate_instruction(sample)
            prompt += self.task_prompt_llasmol.format(input_smiles=input_smiles, opt_prompt=opt_prompt) + "\n"

        if prompt_type == 'ins':
            prompt += self.instructions[sample['instr_idx']] + "\n"
        if prompt_type == 'ins' or prompt_type == 'icl':
            prompt += f"%%% Input : <SMILES> {input_smiles} </SMILES>\n" + f"%%% Adjust: "
            prompt += self.generate_instruction(sample) + "\n"
        if add_response:
            if not output_smiles:
                raise ValueError("Output SMILES must be provided for prompts with response")
            prompt += f"%%% Response: <SMILES> {output_smiles} </SMILES>\n\n"
        elif prompt_type == 'ins':
            prompt += f"%%% Response:\n"

        return prompt

    
    def generate_prompt_for_general_purpose_LLMs(self,
                                                test_file: str,
                                                icl_file: str,
                                                task: str,
                                                prompt_type: str,
                                                model_id: str,
                                                sampling: str,
                                                num_shots: int,
                                                prompt_explain: bool,
                                                instr_setting: str = 'seen'
                                                ) -> List[str]:
        """
        Generates prompts for the given test sample with in-context examples for prompting general-purpose LLMs
        Args:
            data_path: The path to the data files
            task: The task for which the prompts are being generated
            prompt_type: The type of prompt to be generated
            model_id: The model ID for which the prompts are being generated
            opt_type: The type of optimization (simple or threshold)
            sampling: The type of sampling to be used for in-context examples
            num_shots: The number of in-context examples to be used
            prompt_explain: Whether to include explanations in the prompts
        """
        test_data = load_dataset("json", data_files=test_file, split='train')
        #icl_data = load_dataset("json", data_files=icl_file, split='train')
        def load_json_gz(file):
            import gzip, json
            with gzip.open(file, 'rt') as f:
                return json.load(f)
        
        test_data = pd.DataFrame(test_data)
        if num_shots:
            icl_data = load_json_gz(icl_file)
            icl_data = pd.DataFrame(icl_data)
            icl_data = icl_data[icl_data['split'] == 'train']

        if 'mixed' not in task:
            test_data = test_data[(test_data['task'] == task) & (test_data['instr_setting'] == instr_setting)]
            if num_shots:
                icl_data = icl_data[icl_data['task'] == task]
        else:
            test_data = test_data[test_data['instr_setting'] == instr_setting]
            if num_shots:
                icl_data = icl_data[icl_data['instr_setting'] == instr_setting]

        #print(len(test_data), len(icl_data))
        test_data = Dataset.from_pandas(test_data)
        #icl_data = Dataset.from_pandas(icl_data)

        prompts = []
        task_prompt = self.task_prompt if prompt_explain else self.task_prompt_noexplain
        system_prompt = f"<<SYS>>\n{self.system_prompt}\n<</SYS>\n" if (model_id == 'llama' or model_id == "mistral") else ""

        for index, sample in enumerate(test_data):
            test_prompt = self.generate_prompt(sample, 
                                                prompt_type, 
                                                add_response=False)
            if num_shots:
                if 'mixed' in task:
                    icl_task_data = icl_data[icl_data['task'] == sample['task']]
                else:
                    icl_task_data = icl_data[icl_data['task'] == task]

            #print(len(icl_task_data))
            # Generate a random sample of in-context examples from the training pairs
            in_context_examples = ""

            if num_shots:
                if sampling == 'random':
                    # randomly sample from list of dicts
                    rand_idx = random.sample(range(len(icl_task_data)), min(num_shots, len(icl_task_data)))
                    sampled_icl_samples = [icl_task_data.iloc[i] for i in rand_idx]
                elif sampling == 'directional':
                    sampled_icl_samples = []
                    for icl_sample in icl_task_data:
                        same_direction = True
                        for prop in sample['properties'].keys():
                            if np.sign(icl_sample['properties'][prop]['change']) != np.sign(sample['properties'][prop]['change']):
                                same_direction = False
                                break
                        if same_direction:
                            sampled_icl_samples.append(icl_sample)
                    
                    if len(sampled_icl_samples) < num_shots:
                        print(f"Test Pair {index}: Number of IC samples with same direction of change is less than {num_shots}")
                        continue
                
                    rand_idx = random.sample(range(len(sampled_icl_samples)), num_shots)
                    sampled_icl_samples = [sampled_icl_samples[i] for i in rand_idx]
                else:
                    raise ValueError(f"Invalid sampling method: {sampling}")
                
                for icl_sample in sampled_icl_samples:
                    icl_prompt = self.generate_prompt(icl_sample, 
                                                       prompt_type, 
                                                       add_response=True)
                    in_context_examples += icl_prompt
            
            # add [INST] tags for general purpose llms
            inst_open_tag = "[INST]\n" if model_id != "llasmol" else ""
            inst_close_tag = "\n[/INST]\n" if model_id != "llasmol" else ""
            response_tag = "%%% Response:"

            if prompt_type == 'pt0':
                prompt = f"{system_prompt}{inst_open_tag}" + in_context_examples + test_prompt + f"{inst_close_tag}{response_tag}\n"
            elif prompt_type == 'icl' and num_shots:
                prompt = f"{system_prompt}{inst_open_tag}{task_prompt}\n\n" + "Examples:\n" + in_context_examples + "Task:\n" + test_prompt + f"{inst_close_tag}{response_tag}\n"
            else:
                prompt = f"{system_prompt}{inst_open_tag}{task_prompt}\n\n" + "Task:\n" + test_prompt + f"{inst_close_tag}{response_tag}\n"
            prompts.append(prompt)
        
        return prompts
                                   

    def get_response(self, output: str) -> str:
        """
        Extracts the response from the output
        """
        response = output.split(self.response_split)[-1].strip()
        return response
