import os
import random

from .evaluator import Evaluator

from langchain.evaluation import load_evaluator
from langchain_community.chat_models import ChatOpenAI, AzureChatOpenAI

from os import getenv
from dotenv import load_dotenv
load_dotenv()
API_BASE = getenv("API_BASE")
API_KEY = getenv("API_KEY")

REGIONS = {
        "gpt-35-turbo-0125": ["canadaeast", "northcentralus", "southcentralus"],
        "gpt-4-0125-preview": ["eastus", "northcentralus", "southcentralus"],
        "gpt-4-turbo-2024-04-09": ["eastus2", "swedencentral"]
    }

class OpenAIEvaluator(Evaluator):
    DEFAULT_MODEL_KWARGS: dict = dict(temperature=0)
    CRITERIA = {"accuracy": """
                Score 1: The answer is completely unrelated to the reference.
                Score 3: The answer has minor relevance but does not align with the reference.
                Score 5: The answer has moderate relevance but contains inaccuracies.
                Score 7: The answer aligns with the reference but has minor omissions.
                Score 10: The answer is completely accurate and aligns perfectly with the reference.
                Only respond with a numberical score"""}

    def __init__(self,
                 model_name: str = "gpt-4-turbo-2024-04-09",
                 model_kwargs: dict = DEFAULT_MODEL_KWARGS,
                 true_answer: str = None,
                 question_asked: str = None,):
        """
        :param model_name: The name of the model.
        :param model_kwargs: Model configuration. Default is {temperature: 0}
        :param true_answer: The true answer to the question asked.
        :param question_asked: The question asked to the model.
        """

        if (not true_answer) or (not question_asked):
            raise ValueError("true_answer and question_asked must be supplied with init.")

        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.true_answer = true_answer
        self.question_asked = question_asked

        api_base = os.getenv("API_BASE")
        if api_base == "":
            api_base = "https://api.openai.com"
        api_key = os.getenv("API_KEY")
        if (not api_key) or (not api_base):
            raise ValueError("API_KEY and API_BASE must be in env for using openai evaluator.")

        self.api_key = api_key
        self.api_base = api_base
        # self.evaluator = ChatOpenAI(api_base=self.api_base,
        #                                  api_key=self.api_key,
        #                                  model=self.model_name,
        #                                  **self.model_kwargs)
        
        region = random.choice(REGIONS[model_name])
        endpoint = f"{api_base}/{region}"
        self.evaluator = AzureChatOpenAI(azure_endpoint=endpoint,
                                         api_key=self.api_key,
                                         azure_deployment=model_name,
                                         api_version="2024-02-01",
                                         **self.model_kwargs)
        

    def evaluate_response(self, response: str) -> int:
        evaluator = load_evaluator(
            "labeled_score_string",
            criteria=self.CRITERIA,
            llm=self.evaluator,
        )

        eval_result = evaluator.evaluate_strings(
            # The models response
            prediction=response,

            # The actual answer
            reference=self.true_answer,

            # The question asked
            input=self.question_asked,
        )

        return int(eval_result['score'])
