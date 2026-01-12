<div align="center">

# HELPFULSUMM: Helpful Personalized Opinion Summarization via Reinforcement Learning from Review Helpfulness Votes

</div>

This repository maintains the code, data, and model checkpoints for the paper *HELPFULSUMM: Helpful Personalized Opinion Summarization via Reinforcement Learning from Review Helpfulness Votes*

We advanced personalized opinion summarization (POS) by leveraging historical user helpfulness
votes on reviews to better model user interests and ground the generation of personalized summary in opinions helpful to the users.

![Helpful_POS](diagram/Helpful_POS_Task.png)

## Installation
Our model was tested under the following dependencies
- python 3 (tested with 3.9)
- transformers (tested with 4.44.2)
- trl (tested with 0.8.0)

We recommend installing using conda and GPU for reasonable runtime. The following will install all dependencies, referenced from Atlas:
```bash
conda create --name helpfulsumm python=3.9
conda activate helpfulsumm
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

We also need some additional packages to run the code. The list of packages is listed in ```requirements.txt```. On the main directory of the repository, run:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
cd evaluation/AlignScore
pip install .
cd ../bleurt
pip install .
```

For computational feasibility, HELPFULSUMM was trained on a 4-bit GPTQ-quantized version of LLM as backbone.
For GPTQ 4bit inference, please install [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa), following installation instruction from [FastChat docs](https://github.com/lm-sys/FastChat/blob/main/docs/gptq.md).

Additionally, please also install [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ):
```bash
pip install auto-gptq 
  --no-build-isolation 
  --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
```

## The HELPFULSUMM Model
We proposed HELPFULSUMM, a reinforcement learning-based model that utilizes user historical helpfulness votes to align with user preference in both Knowledge Consistency and Persona Consistency.
HELPFULSUMM is trained in two stages following the standard RLHF training paradigm.

[//]: # (- **Stage 1: Supervised Fine-tuning**: First obtain a base summarizer **HelpfulSumm-FT** )

[//]: # (by instruction-finetuning an LLM to equip it with the capability to perform the POS task in end-to-end manner.)

[//]: # ()
[//]: # ([//]: # &#40;with Chain-of-Thought &#40;CoT&#41; prompt&#41;)
[//]: # ([//]: # &#40;that guides the model to identify &#40;1&#41; the user profile and &#41;)
[//]: # ([//]: # &#40;&#40;2&#41; profile-conditioned helpful key points &#40;KPs&#41; during summary generation&#41;)
[//]: # (- **Stage 2: Reinforcement Learning**: we further optimize HelpfulSumm-FT )

[//]: # (via reinforcement learning of the generated summary output using 2 reward models:)

[//]: # (  - **Helpful Opinion Reward:** Use a fine-tuned DeBERTa model to estimate user helpfulness scores &#40;0-5&#41;, i.e., averaged votes, on opinions captured in the summary based on historical user-voted reviews )

[//]: # (  - **Persona Alignment Reward:** Use an LLM to infer the user profile from historical user-voted reviews and score the alignment of the summary to each profile's characteristic accordingly)

### Stage 1: Supervised Finetuning
We first obtain a base summarizer (`HelpfulSumm-FT`) by instruction-finetuning an LLM to equip it with the capability to perform the POS task in end-to-end manner.

To train `HelpfulSumm-FT` with `Llama-3.1-8B-Instruct` as backbone default hyperparameters and settings mentioned in the paper, run the following command:

```
sh run_stage_1_sft.sh
```

*Note: Due to computational limitation, for runtime feasibility, we opt to use [quantized version](https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4) (4 bit, group size 128) of `Llama-3.1-8B-Instruct`. Training a 4-bit quantized `Llama-3.1-8B-Instruct` requires 24GB GPU memory (a RTX 4090 GPU)*

To customize the training setting, please access the file [`run_stage_1_sft.sh`](run_stage_1_sft.sh) and adjust the following arguments:
- `model_name_or_path`: the backbone LLM of HELPFULSUMM instruction fine-tuned for the POS task
- `data_path`: path to SFT training data
- `model_max_length`: max sequence length for training, increase if GPU memory allows

The trained summarizer is saved under [`models/stage_1_helpfulsumm_ft`](models/stage_1_helpfulsumm_ft)

### Stage 2: Reinforcement Learning
We further optimize `HelpfulSumm-FT` via reinforcement learning of the generated summary output using 2 reward models. The RL-optimized model is referred to as `HelpfulSumm-RL`:
- **Helpful Opinion Reward:** Use a fine-tuned DeBERTa model to estimate user helpfulness scores (0-5), i.e., averaged votes, on opinions captured in the summary based on historical user-voted reviews 
- **Persona Alignment Reward:** Use an LLM to infer the user profile from historical user-voted reviews and score the alignment of the summary towards each of the six profile characteristics (e.g., personality traits) accordingly

To train `HelpfulSumm-RL` with `Llama-3.1-8B-Instruct` as backbone default hyperparameters and settings mentioned in the paper, run the following command:
```
export OPENAI_API_KEY="<YOUR-API-KEY>"
sh run_stage_2_rl.sh
```
**IMPORTANT**: Please provide your API key to OpenAI via `<YOUR-API-KEY>` in above command, as the training will utilize OpenAI's LLMs to extract KPs, i.e., opinions, from the generated as well as scoring persona alignment reward. 

To customize the training setting, please access the file [`run_stage_2_rl.sh`](run_stage_2_rl.sh) and adjust the following arguments:
- `policy_base_model`: the backbone LLM of `HelpfulSumm-FT` base summarizer for RL training at this stage
- `policy_model_path`: the checkpoint of `HelpfulSumm-FT` base summarizer for RL training
- `helpful_opinion_reward_base_model`: the encoder base model used in *Helpful Opinion Reward*
- `helpful_opinion_reward_model_path`: the fine-tuned BERT encoder for predicting the helpfulness in the generated summary against user reviews history in Helpful Opinion Reward
- `persona_alignment_reward_model`: the LLM (from OpenAI) used for scoring the persona alignment of generated summary with user profile in *Persona Alignment Reward*
- `data_path`: path to RL training data

The model can also be trained using the [`train/stage_2_rl.ipynb`](train/stage_2_rl.ipynb) notebook 

![HelpfulSumm_Model](diagram/HelpfulSumm_RL_Model.png)

All prompts are located under [```/prompts```](/prompts)

[//]: # (#### Fine-tuning a DeBERTa model for Helpful Opinion Reward)

### Model Checkpoint
For ease of reproducibility, we provided the trained model checkpoints of HELPFULSUMM, using quantized version (4 bit, group size 128) of `Llama-3.1-8B-Instruct` (due to computational limitation for training).  
Model checkpoint can be downloaded from this [Google Drive link](https://drive.google.com/file/d/12dZLHgrChs9rHcog8qpr0LntlxicTrmt/view?usp=sharing).
Please download the file and unzip the ```/models``` directory into the main working directory.

### Inference

## The CiaoHelpful Dataset
We proposed CiaoHelpful, a new dataset specialized for training and evaluation of end-to-end models for helpful personalized opinion summarization.
CiaoHelpful annotates gold summary of product personalized for a user in 5 stages:
- *Stage 1 - KP Extraction from User Reviews:* Extract key points, short i.e., short salient sentences representing review opinions, from reviews voted or written by the user on the product (as gold source of preferred knowledge)  
- *Stage 2 - KP Helpfulness Score Calculation:* Calculate helpfulness scores at the KP (opinion level), by matching extracted KPs (Stage 1) with user-voted reviews and calculate the average helpfulness votes of matching reviews.
- *Stage 3 - User Profile Generation:* Infer the user profile based on their historical voted/written-reviews from similar products
- *Stage 4 - Helpful KP Filtering & Ranking:* Select KPs with high helpfulness score (Stage 2) and make sure they also aligns with the user profile (Stage 3) as gold helpful KPs to the user
- *Stage 5 - Personalized Summary Annotation:* Human-in-the-loop feedback for iterative refinement of gold personalized summary annotated by LLM

![CiaoHelpful_Annotation](diagram/CiaoHelpful_Annotation.png)

The dataset can be accessed under the [```data/```](/data) folder, 
following the [```sft_train/```](/data/sft_train), [```rl_train/```](/data/rl_train) and [```test/```](/data/test) subdirectories for the train and test set.
Files in each sub-directory:
```
data
├── sft_train
│   ├── train.jsonl
│   ├── train.csv
│   ├── train.pkl
├── test
│   ├── test.jsonl
│   ├── test.csv
│   ├── test.pkl
```

Full dataset can be downloaded from this [Google Drive link](https://drive.google.com/file/d/1e3JKCbU98bXETWJesevOIgXICxGxYKa9/view?usp=sharing).
