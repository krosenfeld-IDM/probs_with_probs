# %%
import json
import numpy as np
import matplotlib.pyplot as plt
import probs_with_probs as pwp
from sklearn.metrics import roc_curve, roc_auc_score

# %%
model = "gpt-4o-2024-05-13"
task_list = pwp.data.task_list[:4]
results = {}
for task in task_list:
    # load {task}.json from pwp.data.results
    with open(pwp.paths.results / f'{model}_{task}.json', 'r') as f:
        data = json.load(f)
    results[task] = data

# %%
plt.figure()
for task in task_list:
    y_true = 1*np.array(results[task]['score'])
    y_score = results[task]['p']
    fpr, tpr, threshold = roc_curve(y_true, y_score,pos_label=1)
    plt.plot(fpr,tpr,label = f"{task.replace("_", " ")} (AUC = {roc_auc_score(y_true, y_score):.2f})")
plt.plot([0, 1], [0, 1], color='k', linestyle='--')
# put legend on the right
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title(f'ROC curve for {model}')
# y_true

# %%
model = "gpt-3.5-turbo-0125"
task_list = pwp.data.task_list[:4]
results = {}
for task in task_list:
    # load {task}.json from pwp.data.results
    with open(pwp.paths.results / f'{model}_{task}.json', 'r') as f:
        data = json.load(f)
    results[task] = data

# %%
plt.figure()
for task in task_list:
    y_true = 1*np.array(results[task]['score'])
    y_score = results[task]['p']
    fpr, tpr, threshold = roc_curve(y_true, y_score,pos_label=1)
    plt.plot(fpr,tpr,label = f"{task.replace("_", " ")} (AUC = {roc_auc_score(y_true, y_score):.2f})")
plt.plot([0, 1], [0, 1], color='k', linestyle='--')
# put legend on the right
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
plt.title(f'ROC curve for {model}')
# y_true

# %%

num_samples = 1000
plt.figure()
y_true = np.random.randint(0,2,num_samples); y_score = np.random.rand(num_samples)
fpr, tpr, threshold = roc_curve(y_true, y_score,pos_label=1)
plt.plot(fpr,tpr,label = f"(AUC = {roc_auc_score(y_true, y_score):.2f})")

y_true = np.random.randint(0,2,num_samples); y_score = np.zeros(num_samples)
y_score[y_true==0] = 0.5*np.random.randn(np.sum(y_true==0))
y_score[y_true==1] = 0.5*np.random.randn(np.sum(y_true==1)) + 0.5
fpr, tpr, threshold = roc_curve(y_true, y_score,pos_label=1)
plt.plot(fpr,tpr,label = f"(AUC = {roc_auc_score(y_true, y_score):.2f})")

y_true = np.ones(num_samples); y_score = np.ones(num_samples)
y_true[0] = 0; y_score[0] = 0
fpr, tpr, threshold = roc_curve(y_true, y_score,pos_label=1)
plt.plot(fpr,tpr,label = f"(AUC = {roc_auc_score(y_true, y_score):.2f})")

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.plot([0, 1], [0, 1], color='k', linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
# y_true

# %%



