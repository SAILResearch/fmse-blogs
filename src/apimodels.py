import os
import pathlib
import time
from enum import Enum

import anthropic
import google.generativeai as genai
import openai
import vertexai
from anthropic import AnthropicVertex
from google.api_core.exceptions import GoogleAPIError
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from together import Together
from vertexai.generative_models import GenerativeModel, SafetySetting

SEED = 0
PROMPT_DIR = pathlib.Path(__file__).parent.parent.parent / "prompts"
if "GOOGLE_PROJECT_ID" not in os.environ:
    print("GOOGLE_PROJECT_ID not found in environment variables")
else:
    PROJECT_ID = os.environ["GOOGLE_PROJECT_ID"]


class Models(Enum):
    CLAUDE_3_HAIKU = "claude-3-haiku@20240307"
    CLAUDE_3_5_SONNET = "claude-3-5-sonnet@20240620"
    GEMINI_1_5_PRO = "gemini-1.5-pro"
    GEMINI_1_5_PRO_VERTEX = "gemini-1.5-pro-002"
    GEMINI_1_5_FLASH = "gemini-1.5-flash-002"
    GEMINI_1_5_FLASH_VERTEX = "gemini-1.5-flash-002"
    GEMINI_EXP_VERTEX = "gemini-experimental"
    GPT_4O = "gpt-4o-2024-05-13"
    GPT_4O_MINI = "gpt-4o-mini-2024-07-18"
    LLAMA_3_70b = "meta-llama/Llama-3-70b-chat-hf"
    QWEN_2_72B = "Qwen/Qwen2-72B-Instruct"


REGION_MAP = {
    Models.CLAUDE_3_HAIKU: "europe-west4",  # "europe-west4", "us-central1"
    Models.CLAUDE_3_5_SONNET: "us-east5",
}


def load_prompt(prompt_name: str):
    fp = PROMPT_DIR / f"{prompt_name}.txt"
    if not fp.exists():
        raise FileNotFoundError(f"{fp} not found!")
    with open(fp, "r") as f:
        return f.read()


class APIModel(object):
    def __init__(self, model_name: str, system_msg: str, temperature: float = 0.0, max_output_token=2048, **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.system_msg = system_msg

    def send_message(self, prompt, **kwargs) -> str:
        raise NotImplementedError("implement the function to send and receive message")


class VertexAI(APIModel):
    def __init__(self, model_name: str, system_msg: str, temperature: float = 0.0, max_output_token=2048):
        super().__init__(model_name, system_msg, temperature, max_output_token)
        self.client = AnthropicVertex(region=REGION_MAP[Models(model_name)], project_id=PROJECT_ID)
        self.max_output_token = max_output_token

    def send_message(self, prompt, attempt=10, sleep=10) -> str:
        for i in range(attempt):
            try:
                message = self.client.messages.create(
                    max_tokens=self.max_output_token,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=self.model_name,
                    temperature=self.temperature,
                    system=self.system_msg,
                )
                response = message.content[0].text
                return response
            except anthropic.RateLimitError:
                time.sleep(sleep)
            except anthropic.InternalServerError as e:
                return f"Exceptional case: anthropic.InternalServerError\n{e}"
            except anthropic.APIError as e:
                return f"Exceptional case: anthropic.APIError\n{e}"

        raise ValueError("Tried {attempt} but still failed.")


class GoogleGenAI(APIModel):
    def __init__(self, model_name: str, system_msg: str, temperature: float = 0.0, max_output_token=2048):
        super().__init__(model_name, system_msg, temperature, max_output_token)
        self.max_output_token = max_output_token

        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.generation_config = {
            "temperature": self.temperature,
            "top_p": 0.95,
            # "top_k": 40,
            "max_output_tokens": self.max_output_token,
            "response_mime_type": "text/plain",
        }
        self.client = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            },
            system_instruction=system_msg,
        )

    def send_message(self, prompt, attempt=999, sleep=30) -> str:
        chat_session = self.client.start_chat(
            history=[
            ]
        )
        for i in range(attempt):
            try:
                return chat_session.send_message(prompt).text
            except GoogleAPIError as e:
                print(f"error:\n{e}")
                print(f"attempt {i} failed, sleep {sleep} seconds...")
                time.sleep(sleep)
        raise ValueError(f"Tried {attempt} but still failed.")


class GoogleVertexAI(APIModel):
    def __init__(self, model_name: str, system_msg: str, temperature: float = 0.0, max_output_token=2048):
        super().__init__(model_name, system_msg, temperature, max_output_token)
        self.max_output_token = max_output_token

        self.generation_config = {
            "max_output_tokens": self.max_output_token,
            "temperature": self.temperature,
            "top_p": 0.95,
            # "top_k": 64,
        }

        self.safety_settings = [
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
            SafetySetting(
                category=SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
            ),
        ]

    def send_message(self, prompt, attempt=999, sleep=30) -> str:
        vertexai.init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel(
            self.model_name,
            system_instruction=[self.system_msg],
        )

        for i in range(attempt):
            try:
                response = model.generate_content(
                    [prompt],
                    generation_config=self.generation_config,
                    safety_settings=self.safety_settings,
                    stream=False,
                )
                return response.text
            except GoogleAPIError as e:
                print(f"error:\n{e}")
                print(f"attempt {i} failed, sleep {sleep} seconds...")
                time.sleep(sleep)
        raise ValueError(f"Tried {attempt} but still failed.")


class OpenAI(APIModel):
    def __init__(
            self,
            model_name: str,
            system_msg: str,
            temperature: float = 0.0,
            max_output_token=2048,
    ):
        super().__init__(model_name, system_msg, temperature, max_output_token)
        self.max_output_token = max_output_token
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=5)

    def send_message(self, prompt) -> str:
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_output_token,
            temperature=self.temperature,
        )
        return response.choices[0].message.content


class TogetherAI(APIModel):
    def __init__(
            self,
            model_name: str = "meta-llama/Llama-3-70b-chat-hf",
            system_msg: str = "",
            temperature: float = 0.7,
            max_output_token=512,
            top_p: float = 0.7,
            top_k: int = 50,
            repetition_penalty: float = 1.0,
            **kwargs
    ):
        super().__init__(model_name, system_msg, temperature, max_output_token)
        self.max_output_token = max_output_token
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))

    def send_message(self, prompt) -> str:
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_output_token,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            stop=["<|eot_id|>"],
            stream=False  # You can toggle streaming based on your needs
        )
        return response.choices[0].message.content


MODEL_MAP = {
    Models.CLAUDE_3_HAIKU: VertexAI,
    Models.CLAUDE_3_5_SONNET: VertexAI,
    Models.GEMINI_1_5_PRO: GoogleGenAI,
    Models.GEMINI_1_5_FLASH: GoogleGenAI,
    # VertexAI
    Models.GEMINI_EXP_VERTEX: GoogleVertexAI,
    Models.GEMINI_1_5_PRO_VERTEX: GoogleVertexAI,
    Models.GEMINI_1_5_FLASH_VERTEX: GoogleVertexAI,
    # OpenAI
    Models.GPT_4O_MINI: OpenAI,
    Models.GPT_4O: OpenAI,
    # TogetherAI
    Models.LLAMA_3_70b: TogetherAI,
    Models.QWEN_2_72B: TogetherAI,
}


def get_model(model: Models, system_msg: str, temperature: float = 0.0, *args, **kwargs):
    return MODEL_MAP[model](model.value, system_msg, temperature, *args, **kwargs)
