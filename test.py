# %%
from inspect_ai.log import read_eval_log

log = read_eval_log("logs/REASONING/2025-08-01T16-47-23-07-00_qa-task_5Nfuy3UAoqQZPP4CyUubak.eval")

# %%
print(log.samples[0].scores['objective'].value)
print(log.samples[0].scores['model_graded_qa'].value)
# %%
