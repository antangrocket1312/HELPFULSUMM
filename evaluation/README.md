## Summary Textual Quality Evaluation ##
* ```Summary_Textual_Quality_Evaluation.ipynb```: Perform lexical and semantic comparison of generated personalized summary with gold annotated personalized summary from our CiaoHelpful dataset (Stage 5 of CiaoHelpful curation)
* ```KP_Textual_Quality_Evaluation_sP_sR_sF1.ipynb```: Perform sP/sR/sF1 set-level evaluation of individual generated KPs with reference helpful KPs from our CiaoHelpful dataset (Stage 4 of CiaoHelpful curation).

## Summary Helpfulness Evaluation ##
* ```SHKP_Helpfulness_Evaluation.ipynb```: Perform **Summary Helpful KP Proportion** evaluation, which computes the proportion of helpful key points (KPs), i.e., opinions, in the generated summary that match reference helpful KPs from our CiaoHelpful dataset (Stage 4 of CiaoHelpful curation).
* ```SHS_Helpfulness_Evaluation.ipynb```: Perform **Summary Helpfulness Score** evaluation, measures the averaged helpfulness score of KPs, i.e., opinions, in a summary. Specifically, for every opinion, we score its helpfulness using our fine-tuned DeBERTa model for **Helpful Opinion** reward
