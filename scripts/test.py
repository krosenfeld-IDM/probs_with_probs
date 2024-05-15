import json
import numpy as np
from openai import OpenAI
from datasets import load_dataset
import probs_with_probs as pwp

client = OpenAI()
# task_list = [pwp.data.task_list[0]]
task_list = pwp.data.task_list[:4]
# model = "gpt-3.5-turbo-0125"
model = "gpt-4o-2024-05-13"
n_samples = 100

# max_size_prompt_len_dict, prompt_question_ids_dict = pwp.data.load_tasks(task_list)

def construct_prompt(q, task):
    prompt = """
    You are an expert in the field of {task}. 

    {question}

    Potential answers:
    (A) {ans1}
    (B) {ans2}
    
    Correctly answer this multiple choice question. Answer only "A" or "B".
    """
    ans_dict = {"R": None, "W": None}
    # right answer
    ans_dict["R"] = q[q["target"]]
    remaining = {"A", "B", "C", "D"} - {q["target"]}
    # pick random element of the set "remaining"
    ans_dict["W"] = q[remaining.pop()]
    # shuffle
    keys = list(ans_dict.keys())
    np.random.shuffle(keys)

    if keys[0] == "R":
        ans_key = "A"
    else:
        ans_key = "B"

    # set formatted_task to be the task with underscores replaced by spaces

    return prompt.format(
        task=task.replace("_", " "),
        question=q["input"],
        ans1=ans_dict[keys[0]],
        ans2=ans_dict[keys[1]],
    ), ans_key


if __name__ == "__main__":
    for subject_name in task_list:
        print(f" ************* on {subject_name} *************")
        res = {"score": [], "logprobs": [], "p":[], "q":[], "i":[]}
        task_data = load_dataset("lukaemon/mmlu", subject_name)
        # train, test, validation datasets, return list with input and target
        for i, q in enumerate(task_data["test"]):
            try:
                prompt, ans = construct_prompt(q, subject_name)
                completions = pwp.openai.get_completion(client, prompt, model=model, top_logprobs=2)
                n_tokens = len(completions.choices[0].logprobs.content)
                msg = completions.choices[0].message.content
                if n_tokens != 1:
                    print(f"Uhoh (n={n_tokens}): {completions.choices[0].message.content}")
                elif msg.lower() not in ['a', 'b']:
                    print(f"Uhoh: {completions.choices[0].message.content}")
                else:
                    res["score"].append(
                        completions.choices[0].message.content.lower() == ans.lower()
                    )
                    res["logprobs"].append(
                        completions.choices[0].logprobs.content[0].logprob
                    )
                    res["p"].append(pwp.utils.linear_prob(res["logprobs"][-1]))
                    res["q"].append(pwp.utils.linear_prob(completions.choices[0].logprobs.content[0].top_logprobs[1].logprob))
                    res["i"].append(i)
            except Exception as e:
                print(f"Error: {e}")
            if i == n_samples:
                break

        # write res to paths.results / f"{subject_name}.json"
        with open(pwp.paths.results / f"{model}_{subject_name}.json", "w") as f:
            json.dump(res, f, indent=4)

    print("done")
