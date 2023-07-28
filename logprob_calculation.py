import numpy as np
import random
import torch
import json
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import sys
import time


# set the random seed for reproducibility
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(args):
    questions = []
    answers = []
    decoder = json.JSONDecoder()

    if args.dataset == "gsm8k": # 初步测试基本只使用 GSM8K 数据集.
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                questions.append(json_res["question"].strip())
                answers.append(json_res["answer"].split("#### ")[-1].replace(",", ""))
    elif args.dataset == "aqua":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                qes = json_res["question"].strip() + " Answer Choices:"

                for opt in json_res["options"]:
                    opt = opt.replace(')', ') ')
                    qes += f" ({opt}"

                questions.append(qes)
                answers.append(json_res["correct"])
    elif args.dataset == "svamp":
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["Body"].strip() + " " + line["Question"].strip()
                a = str(line["Answer"])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "asdiv":
        with open(args.dataset_path) as f:
            json_data = json.load(f)["Instances"]
            for line in json_data:
                q = line['input'].strip()
                a = line['output'][0]
                questions.append(q)
                answers.append(a)
    elif args.dataset in ("addsub", "singleeq", "multiarith"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            for line in json_data:
                q = line["sQuestion"].strip()
                a = str(line["lSolutions"][0])
                if a[-2:] == ".0":
                    a = a[:-2]
                questions.append(q)
                answers.append(a)
    elif args.dataset == "csqa":
        with open(args.dataset_path) as f:
            lines = f.readlines()
            for line in lines:
                json_res = decoder.raw_decode(line)[0]
                choice = "Answer Choices:"
                for c in json_res["question"]["choices"]:
                    choice += " ("
                    choice += c["label"]
                    choice += ") "
                    choice += c["text"]
                questions.append(json_res["question"]["stem"].strip() + " " + choice)
                answers.append(json_res["answerKey"])
    elif args.dataset == "strategyqa":
        if 'task' in args.dataset_path:
            with open(args.dataset_path) as f:
                json_data = json.load(f)["examples"]
                for line in json_data:
                    q = line["input"].strip()
                    a = int(line["target_scores"]["Yes"])
                    if a == 1:
                        a = "yes"
                    else:
                        a = "no"
                    questions.append(q)
                    answers.append(a)
        else:
            with open(args.dataset_path, encoding='utf-8') as f:
                json_data = json.load(f)
                for line in json_data:
                    q = line["question"].strip() 
                    if line['answer']:
                        a = 'yes'
                    else:
                        a = 'no'
                    questions.append(q)
                    answers.append(a)
    elif args.dataset in ("coin_flip", "last_letters"):
        with open(args.dataset_path) as f:
            json_data = json.load(f)
            json_data = json_data["examples"]
            for line in json_data:
                q = line["question"]
                a = line["answer"]
                questions.append(q)
                answers.append(a)
    else:
        raise NotImplementedError

    print(f"dataset: {args.dataset}")
    print(f"dataset_size: {len(answers)}")
    args.dataset_size = len(answers)
    return questions, answers


# return a customized dataloader of batches
# Not PyTorch dataloader, it supprts random index(slice) access
def create_dataloader(args)->list:
    set_random_seed(args.random_seed)
    questions, answers = load_data(args)
    dataset = []
    for idx in range(len(questions)):
        dataset.append({"question":questions[idx], "answer":answers[idx], "question_idx":idx})

    random.shuffle(dataset)
    print(f"dataloader size: {len(dataset)}")
    return dataset


# 从 args.prompt_path(一般是 basic_cot_prompts/math_word_problems) 里面读取 Validation Set 所需要的 (q,a)
# return a string of prefix prompt before each question;
# "rationale" is not included in the prefix prompt, only "question" and "pred_ans"
def create_input_prompt(args)->str:
    '''从 args.prompt_path 读取8条(q,a), 用来创建 Validation Set.
    对于 strategy qa 数据集可能不适用.'''
    x, y = [], []
    
    with open(args.prompt_path, encoding="utf-8") as f:
        json_data = json.load(f)
        json_data = json_data["prompt"]
        for line in json_data:
            x.append(line["question"])
            y.append(line["pred_ans"])

    index_list = list(range(len(x)))
    
    prompt_text = ""
    for i in index_list:
        prompt_text += x[i] + " " + y[i] + "\n\n"   # 回答语句应该是 A: xxx, 不要在 y[i] 后面加句号.
    return prompt_text


def generate_logprob_qes(args, qes, with_validation: bool):
    '''返回 logprob 标量值;
    TODO: 检查 output_path, 并把先前已经选出的 (x_0, y_0) 作为 context 接入 prompt.'''
    if with_validation:
        prompt_text = create_input_prompt(args)
        prompt_text += qes["question"] + "\nA: " + qes["answer"]    # No `.` after answer
        logprob = calculate_logprob_ans(args.model, prompt_text)
    else:
        prompt_text = qes["question"] + "\nA: " + qes["answer"]
        logprob = calculate_logprob_ans(args.model, prompt_text)
    
    return logprob


def calculate_logprob_ans(model_path, input_prompt):
    '''Given prompt and model_name, return log-probability of the answer in prompt.'''
    model = AutoModelForCausalLM.from_pretrained(model_path, use_auth_token="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", device_map="auto") # 已经自动做了 device_map, 那就不需要 .to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token="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", trust_remote_code=True, use_fast=True)    # model_max_length=sys.maxsize
    encodings = tokenizer(input_prompt, return_tensors="pt").to("cuda")
    input_ids = encodings["input_ids"]
    assert input_ids.numel() <= 4096, "input_prompt is too long for the model, should discard some contexts."
    labels = input_ids
    
    with torch.no_grad():
        out_logits = model(input_ids).logits
    
    answer_logit = out_logits[:, -2, :] # For prediction scores, answer is at position -2
    answer_label = labels[:, -1]    # For input_ids, answer is at position -1
    loss = torch.nn.CrossEntropyLoss()
    log_prob = - loss(answer_logit, answer_label)
    
    return log_prob


def create_logdifference(args, questions):
    '''The argument provided for `questions` is `dataloader`'''
    result = []
    count = 0

    for qes in questions:
        if count == args.qes_limit:
            break
        logprob_with_validation = generate_logprob_qes(args, qes, with_validation=True)
        logprob = generate_logprob_qes(args, qes, with_validation=False)
        log_difference = logprob_with_validation - logprob
        result.append({
            "dataset_idx": qes["question_idx"],
            "log_difference": log_difference
        })
        count += 1
    
    # Now sort the results by log_difference from big to small
    result.sort(key=lambda x: -x["log_difference"])

    return result


def main():
    args = arg_parser()
    print('*****************************')
    print(args)
    print('*****************************')
    
    set_random_seed(args.random_seed)
    
    dataloader = create_dataloader(args)
    
    if args.dataset_size > 1000:
        dataloader = dataloader[:1000] # only take 1000 questions as selection scope, randomness decided by seed
    print(f"Dataloader size: {len(dataloader)}")

    if args.qes_limit == 0:
        args.qes_limit = len(dataloader)
    
    start = time.time()
    result = create_logdifference(args, dataloader)
    end = time.time()
    print('Total Execution Time: ', end - start, " seconds")
    
    # output the results
    path = f"{args.output_dir}/logdifference_result_{args.dataset}_total_num_{args.qes_limit}.txt"
    with open(path, 'w') as f:
        try:
            f.write(json.dumps(result, indent=4))
        except:
            pass


def arg_parser():
    parser = argparse.ArgumentParser(description="logdifference_calculation")
    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="gsm8k", choices=["gsm8k"], help="dataset to inference"
    )   # choices=["gsm8k","svamp", "aqua", "csqa", "last_letters", "strategyqa", "asdiv", "singleeq", "addsub", "multiarith"]
    parser.add_argument(
        "--prompt_path", type=str, default="./validation_prompts/math_word_problems", help="prompts used to create Validation Set"
    )
    parser.add_argument(
        "--model", type=str, default="baichuan-inc/Baichuan-13B-Chat", help="HuggingFace model used to calculate logprob"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./logdifference_results", help="output directory for logdifference results"
    )
    parser.add_argument(
        "--qes_limit", type=int, default=10, help="whether to limit the size of training set. if 0, the training set is unlimited and we examine all the samples in the dataloader."
    )
    
    args = parser.parse_args()
    
    # Fill in the dataset path
    if args.dataset == "gsm8k":
        args.dataset_path = "./dataset/GSM8K/train.jsonl" # train data path
    elif args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/train.json" # train data path
    elif args.dataset == "csqa":
        args.dataset_path = "./dataset/CSQA/train_rand_split.jsonl" # train data path
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/strategyQA/train.json" # train data path
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters_train2.json" # train data path
    else:
        raise ValueError("dataset is not properly defined ...")
    
    return args