import anthropic
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

client = anthropic.Client(api_key="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")


sample_tag = """
## 1. Problem statement

James is building a hall of mirrors. Three of the walls will be completed covered with glass. If <fact1>two of those walls are 30 feet by 12 feet</fact1> and <fact2>the third is 20 feet by 12 feet</fact2>, how many square feet of glass does he need?

## 2. Facts

<fact1>Two walls dimensions: 30 feet by 12 feet</fact1>
<fact2>Third wall dimensions: 20 feet by 12 feet</fact2>

## 3. Steps

<step1> First find the area of one of the long walls </step1>
<formula> length × width </formula>
<calculation> 30 feet * 12 feet = 360 square feet </calculation>
<var> long_wall_area : 360 square feet </var>

<step2> Then double that amount since there are two walls </step2>
<formula> long_wall_area × 2 </formula>
<calculation> 360 square feet * 2 = 720 square feet </calculation>
<var> two_long_walls_area : 720 square feet </var>

<step3> Then find the area of the short wall </step3>
<formula> length × width </formula>
<calculation> 20 feet * 12 feet = 240 square feet </calculation>
<var> short_wall_area : 240 square feet </var>

<step4> Then add that area to the area of the two long walls to find the total area </step4>
<formula> two_long_walls_area + short_wall_area </formula>
<calculation> 720 square feet + 240 square feet = 960 square feet </calculation>
<var> total_glass_area : 960 square feet </var>

## 4. Wrongsteps

<wrongstep> 0 </wrongstep>

## 5. Output

<output> 960 </output>
"""

class Tag_Generation:
    def __init__(self, problem_statement:str, wrong_step:int , raw_cot:str , problem_type: str,  tag_sample: str ):
        self.problem_statement = problem_statement
        self.problem_type = problem_type
        self.wrongstep = wrong_step
        self.raw_cot = raw_cot
        self.tag_sample = tag_sample
        self.line_list = self.raw_cot.split("\n")
    def get_prompt(self):
        prompt = f"""
            You are given a {self.problem_type} problem: {self.problem_statement}
            You are also given my answer, which you need refer to and must not attempt to verify or change, the answer is composed of different steps and the steps have been divided and put into this list: {self.line_list[:-1]}
            Important:there might be some mistakes in my answer, but please do not try to fix it, leave it as it is and pretend it is true. Do not mention it anywhere in the solution.
            The wrong step for this questions is Step:{self.wrongstep}
            Your task is only to re-format my answer according to the four sections below. Do not solve the problem yourself and do not correct any possible mistakes.
            1. Problem statement
            State the problem. Tag every numeric fact with <fact1> ... </fact1>, <fact2>...</fact2>, etc. Use a tag only for information that contains a number.
            For example: <fact1> Cost of shirts: $50 </fact1>
            2. Facts
            List each parsed fact as a variable description, re-using the same tags. *Example* (do **not** change): > Facts: > <fact1>Starting sapphires: 8</fact1> > <fact2>Sapphires traded: 3</fact2> > …  Make sure only numeric value is allowed on the right side of the colon within the fact tags. 
            3.Steps
            My answer is already divided into steps and put into a list. Provide exactly the same number of steps in this section as the number of steps provided in the list. For each step write:
            <stepX> concise description of the step </stepX>  (do not display any calculation or formula between stepX tags, it should be purely description of the step.)
            <formula> formula shown, blank if none </formula>
            <calculation> numbers shown, blank if none. The calculation need to stay consistent with the information in the description at that step. Do not try to change the number or fix the number</calculation>
            <var> intermediate variable </var>  the number variable in this step depends on how many variables are needed, make sure for all the variable the same tag <var>... </var> is used. Also use consistent variable names across steps. Example: <var>t_shirt_income : $10</var>
            4.Wrongsteps
            This discribe the wrong step that appears in the explanation steps. it was being provided earlier in the prompt. it should only contain the integer of the wrong step with tag e.g.  <wrongstep> 2 </wrongstep>
            5.Output
            The final answer is {self.line_list[-1].lstrip('#')}, copy the final numerical result from my answer and put it inside <output> tags, for example <output>600</output>. 
            #############
            Here is a tag sample:{self.tag_sample}
            """
        return prompt

# helper function to add the step number and error type on the dataframe
def add_step_num_error_type(df, error_type:str):
    transformed_solutions = df['transformed_solution']
    step_num_lst = [len(solution.split('\n'))-1 for solution in transformed_solutions]
    df['num_steps'] = step_num_lst
    df['error_type'] = [error_type]*len(df)
    return df


if __name__ == "__main__":
    ###### load data Calculation Error#########
    df_CA = pd.read_json('cleaned_json_input/CA/generated_cases_clean.jsonl', lines=True)
    # transformation (adding the number of steps and the error type)
    df_CA_step_added = add_step_num_error_type(df_CA,'CA')
    df_CA_transformed = df_CA_step_added[df_CA_step_added['num_steps'].notna()].sort_values(by='num_steps',ascending=False).head(50).sample(frac=1, random_state=42).reset_index(drop = True)
    df_CA_transformed['internal_num'] = list(range(1, 51))

    ###### load data Counting Error#########
    df_CO= pd.read_json('cleaned_json_input/CO/generated_cases_clean.jsonl', lines=True)
    # transformation (adding the number of steps and the error type)
    df_CO_step_added = add_step_num_error_type(df_CO,'CO')
    df_CO_transformed = df_CO_step_added[df_CO_step_added['num_steps'].notna()].sort_values(by='num_steps',ascending=False).head(50).sample(frac=1, random_state=42).reset_index(drop = True)
    df_CO_transformed['internal_num'] = list(range(1, 51))

    ###### load data Contradictory Step#########
    df_CS= pd.read_json('cleaned_json_input/CS/generated_cases_clean.jsonl', lines=True)
    # transformation (adding the number of steps and the error type)
    df_CS_step_added= add_step_num_error_type(df_CS,'CS')
    df_CS_transformed = df_CS_step_added[df_CS_step_added['num_steps'].notna()].sort_values(by='num_steps',ascending=False).head(50).sample(frac=1, random_state=42).reset_index(drop = True)
    df_CS_transformed['internal_num'] = list(range(1, 51))


    ###### load data Context Value Error#########
    df_CV= pd.read_json('cleaned_json_input/CV/generated_cases_clean.jsonl', lines=True)
    # transformation (adding the number of steps and the error type)
    df_CV_step_added = add_step_num_error_type(df_CV,'CV')
    df_CV_transformed = df_CV_step_added[df_CV_step_added['num_steps'].notna()].sort_values(by='num_steps',ascending=False).head(50).sample(frac=1, random_state=42).reset_index(drop = True)
    df_CV_transformed['internal_num'] = list(range(1, 51))


    ###### load data Formula Confusion Error#########
    df_FC= pd.read_json('cleaned_json_input/FC/generated_cases_clean.jsonl', lines=True)
    # transformation (adding the number of steps and the error type)
    df_FC_step_added = add_step_num_error_type(df_FC,'FC')
    df_FC_transformed = df_FC_step_added[df_FC_step_added['num_steps'].notna()].sort_values(by='num_steps',ascending=False).head(50).sample(frac=1, random_state=42).reset_index(drop = True)
    df_FC_transformed['internal_num'] = list(range(1, 51))


    ###### load data Hallucination Error#########
    df_HA= pd.read_json('cleaned_json_input/HA/generated_cases_clean.jsonl', lines=True)
    # transformation (adding the number of steps and the error type)
    df_HA_step_added  = add_step_num_error_type(df_HA,'HA')
    df_HA_transformed = df_HA_step_added[df_HA_step_added['num_steps'].notna()].sort_values(by='num_steps',ascending=False).head(50).sample(frac=1, random_state=42).reset_index(drop = True)
    df_HA_transformed['internal_num'] = list(range(1, 51))


    ###### load data Missing Step Error#########
    df_MS= pd.read_json('cleaned_json_input/MS/generated_cases_clean.jsonl', lines=True)
    # transformation (adding the number of steps and the error type)
    df_MS_step_added = add_step_num_error_type(df_MS,'MS')
    df_MS_transformed = df_MS_step_added[df_MS_step_added['num_steps'].notna()].sort_values(by='num_steps',ascending=False).head(50).sample(frac=1, random_state=42).reset_index(drop = True)
    df_MS_transformed['internal_num'] = list(range(1, 51))


    ###### load data Operator Error#########
    df_OP= pd.read_json('cleaned_json_input/OP/generated_cases_clean.jsonl', lines=True)
    # transformation (adding the number of steps and the error type)
    df_OP_step_added = add_step_num_error_type(df_OP,'OP')
    df_OP_transformed = df_OP_step_added[df_OP_step_added['num_steps'].notna()].sort_values(by='num_steps',ascending=False).head(50).sample(frac=1, random_state=42).reset_index(drop = True)
    df_OP_transformed['internal_num'] = list(range(1, 51))


    ###### load data Unit Conversion Error#########
    df_UC= pd.read_json('cleaned_json_input/UC/generated_cases_clean.jsonl', lines=True)
    # transformation (adding the number of steps and the error type)
    df_UC_step_added = add_step_num_error_type(df_UC,'UC')
    df_UC_transformed = df_UC_step_added[df_UC_step_added['num_steps'].notna()].sort_values(by='num_steps',ascending=False).head(50).sample(frac=1, random_state=42).reset_index(drop = True)
    df_UC_transformed['internal_num'] = list(range(1, 51))


    # combine the dataframe:
    error_frames = [df_CA_transformed, df_CO_transformed, df_CS_transformed, df_CV_transformed, df_FC_transformed, df_HA_transformed, df_MS_transformed, df_OP_transformed, df_UC_transformed]

    combined_frames = pd.concat(error_frames).reset_index(drop=True)
    combined_frames['wrong_type'] = "wrong"

    df_right = combined_frames.sample(n=50,random_state=42)
    df_right['wrong_step'] = 0
    df_right['wrong_type'] = "right"
    df_right['error_type'] = "NA"
    df_right['internal_num'] = list(range(1, 51))

    total_frames = [combined_frames, df_right]
    total_combined_frames = pd.concat(total_frames).reset_index(drop=True)

    # check the distribution of the errores: 
    category_counts = total_combined_frames['error_type'].value_counts()
    print(category_counts)
    
    # wrong questions: 
    for i in range(len(total_combined_frames)):
        if total_combined_frames['wrong_type'][i] == 'wrong':
            tag_generation = Tag_Generation(problem_statement = total_combined_frames['question'][i], wrong_step = total_combined_frames['wrong_step'][i],  raw_cot = total_combined_frames['transformed_solution'][i], problem_type = "GSM8K", tag_sample = sample_tag)
        else:
            tag_generation = Tag_Generation(problem_statement = total_combined_frames['question'][i], wrong_step = total_combined_frames['wrong_step'][i],  raw_cot = total_combined_frames['original_solution'][i], problem_type = "GSM8K", tag_sample = sample_tag)
        tag_prompt = tag_generation.get_prompt()
        with open(f"tags/tag_generation_{total_combined_frames['wrong_type'][i]}_{total_combined_frames['error_type'][i]}_{total_combined_frames['internal_num'][i]}.txt", "w", encoding="utf-8") as f:
            with client.messages.stream(
                model="claude-3-7-sonnet-20250219",
                max_tokens=64000,
                temperature=0.2,
                messages=[{"role": "user", "content": tag_prompt}],
            ) as stream:    
                for text in stream.text_stream:
                    f.write(text) 
     
    # right explanation for the CA error:  
    for i in range(50):
        tag_generation = Tag_Generation(problem_statement = total_combined_frames['question'][i], wrong_step = total_combined_frames['wrong_step'][i],  raw_cot = total_combined_frames['original_solution'][i])
        tag_prompt = tag_generation.get_prompt()
        with open(f"tags/tag_generation_right_{total_combined_frames['error_type'][i]}_{total_combined_frames['internal_num'][i]}.txt", "w", encoding="utf-8") as f:
            with client.messages.stream(
                model="claude-3-7-sonnet-20250219",
                max_tokens=64000,
                temperature=0.2,
                messages=[{"role": "user", "content": tag_prompt}],
            ) as stream:    
                for text in stream.text_stream:
                    f.write(text) 
