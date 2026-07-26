# for system prompt
think_prompt = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

# for strategyQA
strategyQA_student_answer_question_prompt = '''
Questions: {}
Let's think step by step. First, you should answer "true" or "false". Then, you should explain how you drew this conclusion.
'''

# for logiQA
logiQA_student_answer_question_prompt = '''
Given the context: {}
Query: {}
There are four options in the list: {}. 
Let's think step by step. First, you should choose one number from [0,1,2,3] and wrap it by [], which represents your answer from the option list. Then, you should explain how you get this answer.
'''

teacher_model_feedback_prompt = '''
Imagine you are a teacher, I will give you one student's incorrect answer to a question. You should point out the mistakes in the student's answer.
Question: {}
Student's Answer: {}
Hint: The correct answer should be {}.
'''

teacher_model_peer_review_prompt = '''
Imagine you are a reviewer, I will give you one submission about the rationale for a question. You should simulate the peer-review process by evaluating the rationale based on its correctness and soundness. Let's think step by step, but your final answer should only be one number, ranging from 1-5 (the higher the score is, the more possible you think the rationale is correct). The score should be wrapped by square brackets, such as: [5].
Question: {}
Rationale: {}
Hint: The correct answer should be {}.
'''

train_feedback_prompt = '''
### Instruction:
Imagine you are a teacher, I will give you one student's incorrect answer to a question. You should point out the mistakes in the student's answer.
### Input:
{}
### Response:
'''

train_rationale_prompt = '''
### Instruction:
Answer the following question. Let's think step by step.
### Input:
{}
### Response:
'''

strategyQA_train_rationale_prompt = '''
### Instruction:
Answer the following question. Let's think step by step. First, you should answer "true" or "false". Then, you should explain how you drew this conclusion.
### Input:
{}
### Response:
'''

logiQA_train_rationale_prompt = '''
### Instruction:
Answer the following question based on the given context, query and options.
### Input:
{}
### Response:
'''

chat_temp = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""}]

if __name__ == "__main__":
    print(think_prompt)