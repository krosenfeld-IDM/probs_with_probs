# List of task we consider
from datasets import load_dataset

task_list = ['college_computer_science', 'formal_logic', 'high_school_computer_science',
             'computer_security', 'machine_learning',

             'clinical_knowledge', 'high_school_biology', 'anatomy', 'college_chemistry',
             'college_medicine', 'professional_medicine',

             'business_ethics', 'professional_accounting', 'public_relations',
             'management', 'marketing'
             ]

def get_prompt(task_data, task, question_num=0, prompt_q=None):
    '''
    task_data:
    Question num specifies which question will be used as prompt.
    If prompt_q is provided, it is used as 1-shot prompt question. This
    corresponds to GPT-4 based question prompts that we created. Else, we
    select question corresponding to question_num from the MMLU itself to
    generate the prompt. We select prompt from test set in this case,
    since train set is very small sometime and may not have 10 samples.
    We use 10 different prompts and take avergae over them to estimate
    performance on a subject. The function returns the 1-shot question prompt.
    '''

    if prompt_q is None:
        prompt_set = 'test'
        if question_num > len(task_data['test']['input']) - 1:
            print('prompt question id exceeds the length of test set')
            print('selecting last question of the test set')
            question_num = len(task_data['test']['input']) - 1
        prompt_add = f'This is a question from {task.replace("_", " ")}.\n'
        prompt_add += f"{task_data[prompt_set]['input'][question_num]}\n"
        for letter in ['A', 'B', 'C', 'D']:
            prompt_add += '    ' + letter + '. ' + task_data[prompt_set][letter][question_num] + '\n'
        prompt_add += f"The correct answer is option: {task_data[prompt_set]['target'][question_num]}\n"
    else:
        prompt_add = f'This is a question from {task.replace("_", " ")}.'
        prompt_add += prompt_q
        prompt_add += '\n'
    prompt_add += f"You are the world's best expert in {task.replace('_', ' ')}. "
    prompt_add += '''Reason step-by-step and answer the following question. '''
    return prompt_add

def get_max_size_prompt_len(task_data, task, n=10, max_allowed_prompt_len=700):
    '''
    get the size of maximum length prompt out of all n prompts considered.
    '''
    max_len = 0
    i = 0
    prompt_question_ids = []
    while len(prompt_question_ids) < n:
        prompt_add = get_prompt(task_data, task=task, question_num=i)
        prompt_len = len(prompt_add)

        if prompt_len > max_allowed_prompt_len:
            i += 1
            continue
        else:
            prompt_question_ids.append(i)
            i += 1

        if prompt_len > max_len:
            max_len = prompt_len
    return max_len, prompt_question_ids


def task_max_prompt_sizes(task_list, max_allowed_prompt_len=1000,n = 10):
    # n is the number of different MMLU based prompts used.
    # Massive Multitask Language Understanding

    max_size_prompt_len_dict = {}
    prompt_question_ids_dict = {}
    for subject_name in task_list:
        task_data = load_dataset('lukaemon/mmlu', subject_name)
        max_len, prompt_question_ids = get_max_size_prompt_len(task_data, subject_name, n=n,
                                                            max_allowed_prompt_len=max_allowed_prompt_len)
        max_size_prompt_len_dict[subject_name] = max_len
        prompt_question_ids_dict[subject_name] = prompt_question_ids

    return  max_size_prompt_len_dict, prompt_question_ids_dict