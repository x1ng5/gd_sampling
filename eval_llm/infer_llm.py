import json
import argparse

from tqdm import tqdm

from sglang import (
    gen,
    function,
    set_default_backend, 
    
    RuntimeEndpoint
)

from utils.utils import(
    read_jsonl,
    get_majority_answer,
    extract_rawdata_answer,
    extract_completion_answer,
    few_shot_prompt, INVALID_ANS,
)
@function
def multi_turn_question(s, prompt, temperature = 1, **kwargs):
    s += prompt
    s += gen("A", max_tokens=512, temperature=temperature,stop=["\nQ:","\n\n"])
    
@function
def multiple_sampling(s,prompt, sampling_num, temperature = 1, **kwargs):
# (s, id, question, sampling_num, max_tokens, ground_truth_answer, temperature, model_type):
    s += prompt
    forks = s.fork(sampling_num)
    forks += gen("A", max_tokens=512, temperature=temperature,stop=["\nQ:","\n\n"])
    answers = []
    for state in forks:
        answers += [state["A"]]
    return answers

parser = argparse.ArgumentParser(description="Process data type.")


parser.add_argument("--num-threads",  default=10,    type=int)
parser.add_argument("--sampling-num", default=10,    type=int)

parser.add_argument("--temperature",  default=1.0,   type=float)

args         = parser.parse_args()

set_default_backend(RuntimeEndpoint("http://localhost:30000"))

gsm_8k_test     = read_jsonl("./data/test.jsonl")
temperature     = args.temperature
num_threads     = args.num_threads
sampling_num    = args.sampling_num
input_data_list = [
    {
        "prompt" : f"{few_shot_prompt}\n\n\nQ: {gsm_8k_item['question']}\nA:", 
        "sampling_num" : sampling_num, 
     } for gsm_8k_item in gsm_8k_test
]
batch_size = 2000
ans_list        = [float(extract_rawdata_answer(gsm_8k_item['answer'])) for gsm_8k_item in gsm_8k_test]
data_all        = []
input_data_list = [input_data_list[idx * 32 : min((idx +1) * 32 , len(input_data_list) - 1)] for idx in range(len(input_data_list)//32 + 1)]
for i, input_data_item in tqdm(enumerate(input_data_list)):
    # input_data_list[i * batch_size : min((i+1) * batch_size, len(input_data_list) - 1)]
    states      = multiple_sampling.run_batch(
                        # [input_data_item],
                        input_data_item,
                        num_threads=num_threads, progress_bar=False
                    )
    data_all   += [state.ret_value for state in states]
    if i % batch_size == batch_size - 1:
        # with open(f"./result/greedy_model_outputs_{i // batch_size}.json", "w+") as fp:
        with open(f"./result/low_T_model_outputs_T={temperature}.json", "w+") as fp:
        
            json.dump(
                data_all,
                fp
            )
        data_all = []
# with open(f"./result/greedy_model_outputs_{i // batch_size + 1}.json", "w+") as fp:
with open(f"./result/low_T_model_outputs_T={temperature}.json", "w+") as fp:
    json.dump(
        data_all,
        fp
    )

# [get_majority_answer([extract_completion_answer(i) for i in state.ret_value ]) for state in states]