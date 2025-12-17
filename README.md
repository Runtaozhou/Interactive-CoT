# Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces

This repository summarizes the paper **Improving Human Verification of LLM Reasoning through Interactive Explanation Interfaces**.

## Overview
Large Language Models often generate long chain-of-thought (CoT) explanations that overwhelm users.  
This project introduces **interactive explanation formats**—iCoT, iPoT, and iGraph—to reduce cognitive load and improve reasoning verification.

### Project Motivation
![Figure 1](Figures/teaser.png)
Given a GSM8K question, LLMs typically provide step-by-step reasoning followed by the final answer. However, such output
presentation is often static and long, posing a higher cognitive load to users and leading to slower and more erroneous answer
verification. In contrast, we prompt LLMs to generate an interactive HTML/JavaScript application wrapped around the reasoning. This
interface enables users to verify the reasoning more efficiently via tools of (a) navigation buttons (inspired by those in common IDEs)
and (b) colored highlights

### Project Pipline
![Figure 2](Figures/overview.png)
Our Explanation Generation pipeline consists of three stages: (A) Tagged Information Generation, where an LLM produces
correct and erroneous GSM8K explanations annotated with reasoning tags; (B) Interface Template Design, where standardized
HTML/CSS templates ensure consistent structure and interactivity across formats; and (C) Explanation Interface Generation, where
tagged data and templates are combined to create interactive explanations in iCoT, iPoT, and iGraph formats for our user study


### Interface Overiew
![Figure 3](Figures/interface_display.png)
Fig. 2. Examples of four explanation formats used in the study: (A) traditional Chain-of-Thought (CoT), (B) interactive Chain-of-
Thought (iCoT), (C) interactive Program-of-Thought (iPoT), and (D) interactive Graph (iGraph). Each format presents the same
reasoning steps in a different modality (textual, structured, code-like, or visual). For consistency, all four formats present the same
mathematical problem. 


## Quantative Results
![Figure 4](Figures/avg_time_by_format.png)
Response time. Results show that participants were able to respond faster using iGraph.
![Figure 5](Figures/avg_verfication_accuracy_by_format.png)
Verification accuracy. Results show that participants achieve the highest verification accuracy using iGraph
![Figure 6](Figures/avg_wrong_step_id_accuracy_by_format.png)
Error localization. Results show that participants were able to detect the exact error step in the LLM’s explanation with higher accuracy using iGraph.


## Qualitative Results
![Figure 7](Figures/survey_result.png)
sPost SurveyQuestionnaire results for comparing different explanation formats using utility questions (G1-G5) and design-based
questions for iPoT, iCoT, and iGraph (D1-D4). Results from the utility questions show that participants find the interactive formats more
effective and engaging, reporting improvements in understanding, error detection, engagement, preference, and overall satisfaction
(G1–G5) compared to the traditional CoT. Similarly, results from the design-based questions indicate that participants across all
interactive formats consider all four design elements helpful during the experiment. The visualization follows methodology from  For the general measures assessment, we find that iGraph (A) achieves the highest rating as compared to CoT (B), iCoT (C).

## Key Takeaways
- Interactive reasoning significantly improves users’ ability to verify LLM reasoning.
- Graph-based explanations outperform standard CoT.
- Structured, interactive interfaces reduce cognitive load.
- The approach generalizes to many reasoning domains.

