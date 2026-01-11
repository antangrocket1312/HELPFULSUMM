import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], './utils'))
import ast
from prompting import *
import torch
import numpy as np
from rouge_score import rouge_scorer


def process_query_tensor(qt):
    i = qt.tolist().index(128000)
    return qt[i:]


def save_check_point():
    pass


def get_step_logs(stats, batch, rewards, step_i):
    logs = {}
    logs['step'] = step_i
    rewards = torch.stack(rewards)
    for k, v in stats.items():
        if not isinstance(v, np.ndarray):
            logs[k] = v
    logs["env/reward_mean"] = torch.mean(rewards).item()  # torch.mean(rewards).cpu().numpy().item()
    logs["env/reward_std"] = torch.std(rewards).item()  # torch.std(rewards).cpu().numpy().item()
    logs["env/reward_dist"] = rewards.tolist()  # rewards.cpu().numpy()
    return logs


def get_text_logs(pred, ans, epoch, step_i, idx_b):
    logs = []
    for p, a, idx in zip(pred, ans, idx_b):
        d = {'epoch': epoch, 'step': step_i, 'id': idx, 'output': p}
        logs.append(d)
        idx += 1
    return logs


def claim_extraction_from_summary(base_prompt, model, text, client):
    claim_split_prompt = base_prompt % (text)
    attempt = 0
    claim_split_response = None
    while attempt < 3 and claim_split_response == None:
        claim_split_response = get_completion(claim_split_prompt, model, client)
        attempt += 1

    if claim_split_response != None:
        claim_split_response = claim_split_response.replace("`", "").replace("json", "")
        generated_summary_kps = ast.literal_eval(claim_split_response)
    else:
        generated_summary_kps = []
    return generated_summary_kps


def calculate_rouge_score(row):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougel_scores = [], [], []
    for rev in row['filtered_hist_vote_written']:
        scores = scorer.score(row['key_point'], rev)
        rouge1_scores += [scores['rouge1'].fmeasure]
        rouge2_scores += [scores['rouge2'].fmeasure]
        rougel_scores += [scores['rougeL'].fmeasure]

    row['rouge1_scores'] = rouge1_scores
    row['rouge2_scores'] = rouge2_scores
    row['rougel_scores'] = rougel_scores

    return row


def filter_hist_vote_written(row):
    my_df = row[['filtered_hist_vote_written', 'rouge1_scores', 'rouge2_scores', 'rougel_scores']].to_frame().T
    my_df = my_df.explode(['filtered_hist_vote_written', 'rouge1_scores', 'rouge2_scores', 'rougel_scores'])
    my_df = my_df.sort_values(by=['rougel_scores', 'rouge2_scores', 'rouge1_scores'], ascending=False)
    row['sorted_hist_vote_written'] = my_df['filtered_hist_vote_written'].tolist()
    return row


def save_rl_data(root_path, kp_df):
    os.makedirs(root_path, exist_ok=True)
    category = kp_df['category'].iloc[0]
    user_id = kp_df['user_id'].iloc[0]
    idx = str(kp_df['id'].iloc[0])
    kp_df.to_pickle(root_path + f"/{category}_{idx}_{user_id}.pkl")