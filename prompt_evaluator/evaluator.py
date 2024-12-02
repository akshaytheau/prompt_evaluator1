from openai import OpenAI
import re

EVALUATION_CRITERIA = {
    "clarity": "How clear and unambiguous is the prompt?",
    "specificity": "Does the prompt specify what is expected in the output?",
    "relevance": "Is the prompt directly related to the intended task?",
    "completeness": "Does the prompt provide all necessary context for the task?",
    "neutrality": "Is the prompt free from bias or leading language?",
    "efficiency": "Is the prompt concise and free of unnecessary verbosity?",
}

class PromptEvaluator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key = api_key)

    def query_model(self, prompt, criterion, question):
        evaluation_question = f"On a scale of 1 to 5, evaluate the following prompt based on {criterion}:\n\n" \
                              f"Prompt: {prompt}\n\n" \
                              f"Question: {question}\n\n" \
                              f"Provide only the numeric score (1-5) and a brief explanation."
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are an assistant that evaluates prompts."},
                      {"role": "user", "content": evaluation_question}],
            temperature=0.2
        )
        return response.choices[0].message

    def evaluate_prompt(self, prompt):
        # scores = {}
        final_response = []
        for criterion, question in EVALUATION_CRITERIA.items():
            response = self.query_model(prompt, criterion, question)
            final_response.append(response.content)
            # try:
            #     score_pattern = r"SCORE:\s*(\d+)"
            #     explanation_pattern = r"EXPLANATION:\s*(.+)"

            #     # Extracting score
            #     score_match = re.search(score_pattern, response.content)
            #     score = score_match.group(1) if score_match else None

            #     # Extracting explanation
            #     explanation_match = re.search(explanation_pattern, response.content)
            #     explanation = explanation_match.group(1) if explanation_match else None
            #     scores[criterion] = {"score": score, "explanation": explanation}
            # except (ValueError, IndexError):
            #     scores[criterion] = {"score": 0, "explanation": "Invalid response from model."}

        # total_score = sum(entry["score"] for entry in scores.values())
        #return scores, total_score
        #print(response)
        return final_response


