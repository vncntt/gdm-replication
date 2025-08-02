from inspect_ai.log import read_eval_log
import json

thinking_log = read_eval_log("../logs/uplift_logs/2025-07-20T20-05-28-07-00_qa-task_JnzwQJwJ9bU4jjTnfq9phj.eval")
immediate_log = read_eval_log("../logs/uplift_logs/2025-07-20T19-51-03-07-00_qa-task_LmzpZJCpWpiURu7RqdKPqv.eval")

thinking_accuracy = {}
immediate_accuracy = {}
for t_sample,i_sample in zip(thinking_log.samples, immediate_log.samples):
    if t_sample.id not in thinking_accuracy:
        thinking_accuracy[t_sample.id] = []
    if i_sample.id not in immediate_accuracy:
        immediate_accuracy[i_sample.id] = []
    else:
        thinking_accuracy[t_sample.id].append(0 if t_sample.scores['model_graded_qa'].value == 'I' else 1) 
        immediate_accuracy[i_sample.id].append(0 if i_sample.scores['model_graded_qa'].value == 'I' else 1) 

thinking_results = {pid: sum(scores)/len(scores) for pid, scores in thinking_accuracy.items()}
immediate_results = {pid: sum(scores)/len(scores) for pid, scores in immediate_accuracy.items()}

combined = {key: [thinking_results[key],immediate_results[key]] for key in thinking_results.keys()}

difference = {key: abs(combined[key][0] - combined[key][1]) for key in combined.keys()}

large_difference_ids = [str(key) for key,diff in difference.items() if diff > 0.5]
print(len(large_difference_ids))

with open("uplift_ids.txt", "w") as f:
    f.write("\n".join(large_difference_ids))

    
problems_data = {}
id_to_sample = {}
for sample in thinking_log.samples:
    id_to_sample[sample.id] = sample

problems_and_answers = {}
for sample in thinking_log.samples:
    if str(sample.id) in large_difference_ids:
        answer = sample.target[0]
        problems_and_answers[sample.id] = {
            'question': sample.input,
            'correct_answer': answer,
        }

with open("extracted_gpqa_extended_problems.json", "w") as f:
    json.dump(problems_and_answers, f, indent=2)