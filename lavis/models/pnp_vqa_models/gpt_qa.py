import time
import json
import openai
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel

def gpt(input_text, model_name, stop_sequences=None, max_length=50, wait_time=0.5, temperature=0.0, logit_bias={}):
    try:
        print(input_text)
        response = openai.Completion.create(
            engine=model_name,
            prompt=input_text,
            temperature=temperature,
            max_tokens=max_length,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequences,
            logit_bias=logit_bias,
        )
        output = response["choices"][0]["text"].strip()
    except (openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError, openai.error.APIConnectionError, openai.error.Timeout, openai.error.InvalidRequestError) as e:
        print('RATE LIMIT ERROR')
        wait_time *= 2
        wait_time = min(wait_time, 5)
        time.sleep(wait_time)
        return gpt(input_text, model_name, stop_sequences, max_length, wait_time)
    return output


@registry.register_model("gpt_qa")
class GPTQA(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {}

    def __init__(
        self,
        model_name,
        num_tries,
        examples_path=None,
        captions_per_image=None,
        temperature=0.0,
        context_separator=" ",
        left_right_images=False,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_tries = num_tries
        self.captions_per_image = captions_per_image
        # PICa prompt
        self.prompt = (
            "Please answer the question according to the above context.\n===\n"
        )
        self.answer_prefix = None
        self.stop_sequences = ["\n", "\r"]
        self.temperature = temperature
        self.context_separator = context_separator
        self.prompt += "Context:" + self.context_separator + "|||1\n===\nQ: |||2\nA:"
        self.left_right_images = left_right_images

    def construct_prompt(self, examples):
        prompt = "Please answer the question according to the above context.\n===\n"
        if "subquestions" in examples[0] or "long_answer" in examples[0]:
            prompt = "Please answer the question according to the above context by reasoning step by step.\n===\n"
        for ex in examples:
            captions = ex["captions"]
            if isinstance(captions[0], list):
                self.context_separator = "\n"
                prompt += "Context:" + self.context_separator
                for i in range(len(captions)):
                    captions_i = captions[i]
                    if self.captions_per_image is not None:
                        captions_i = captions_i[: self.captions_per_image]
                    prompt += (
                        "Image "
                        + str(i + 1)
                        + ": "
                        + " ".join(
                            [cap[0].upper() + cap[1:] + "." for cap in captions_i]
                        )
                        + "\n"
                    )
                prompt += (
                    "===\nQ: "
                    + ex["question"][0].upper()
                    + ex["question"][1:].lower()
                    + "\n"
                )
                prompt += "A: " + ex["answer"] + "\n\n===\n"
            else:
                if self.captions_per_image is not None:
                    captions = captions[: self.captions_per_image]
                prompt += (
                    "Context:"
                    + self.context_separator
                    + " ".join([cap[0].upper() + cap[1:] + "." for cap in captions])
                    + "\n===\n"
                    + "Q: "
                    + ex["question"][0].upper()
                    + ex["question"][1:]
                    + "\n"
                )
                if "subquestions" in ex:
                    prompt += "Are subquestions needed here: "
                    if len(ex["subquestions"]) > 0:
                        prompt += "yes"
                    else:
                        prompt += "no"
                    prompt += "\n"
                    for subquestion in ex["subquestions"]:
                        prompt += (
                            "Subquestion: "
                            + subquestion["question"][0].upper()
                            + subquestion["question"][1:]
                            + "\n"
                        )
                        prompt += "Intermediate Answer: " + subquestion["answer"] + "\n"
                if "long_answer" in ex:
                    prompt += "A: " + ex["long_answer"] + "\n\n===\n"
                elif "subquestions" in ex:
                    prompt += "One-word final answer: " + ex["answer"] + "\n\n===\n"
                else:
                    prompt += "A: " + ex["answer"] + "\n\n===\n"
        if "subquestions" in examples[0]:
            prompt += (
                "Context:"
                + self.context_separator
                + "|||1\nQ: |||2\nAre subquestions needed here:"
            )
            prompt = self.prompt.replace("===\n", "")
            # self.answer_prefix = "So the final answer is:"
            # self.answer_prefix = "A:"
            self.answer_prefix = "One-word final answer:"
            self.stop_sequences = ["\n\n", "\r\r", "\n\r", "\r\n"]
        elif "long_answer" in examples[0]:
            # self.answer_prefix = "So the final answer is:"
            prompt += (
                "Context:"
                + self.context_separator
                + "|||1\nQ: |||2\nA: Let's think step by step."
            )
            self.answer_prefix = "Final one word answer:"
            self.stop_sequences = ["\n\n", "\r\r", "\n\r", "\r\n"]
        else:
            # self.prompt += "Context: |||1\n===\nQ: |||2\nA:"
            prompt += "Context:" + self.context_separator + "|||1\n===\nQ: |||2\nA:"
        return prompt

    def generate(self, question, captions, prompt=None, max_length=20):
        captions_to_use = captions
        if isinstance(captions[0], list):
            evidence = ""
            for i in range(len(captions)):
                captions_to_use = captions[i]
                if self.captions_per_image is not None:
                    captions_to_use = captions_to_use[: self.captions_per_image]
                evidence += (
                    "Image "
                    + str(i + 1)
                    + ": "
                    + " ".join(
                        [
                            cap[0].upper() + cap[1:].lower() + "."
                            if cap[-1] != "."
                            else cap[0].upper() + cap[1:].lower()
                            for cap in captions_to_use
                        ]
                    )
                    + "\n"
                )
            evidence = evidence[:-1]
        else:
            if self.captions_per_image is not None:
                captions_to_use = captions[: self.captions_per_image]
            evidence = " ".join(
                [
                    cap[0].upper() + cap[1:].lower() + "."
                    if cap[-1] != "."
                    else cap[0].upper() + cap[1:]
                    for cap in captions_to_use
                ]
            )
        # input_text = "Answer the question about the picture based on the captions.\nCaptions:\n"+evidence+"\nQuestion: "+question+"\nAnswer (1 or 2 words):"
        if prompt is None:
            prompt = self.prompt
        input_text = prompt.replace("|||1", evidence).replace(
            "|||2", question[0].upper() + question[1:]
        )
        if self.left_right_images:
            input_text = input_text.replace("Image 1:", "Left image:").replace("Image 2:", "Right image:")
        output = ""
        num_tries = 0
        while len(output) == 0 and num_tries < self.num_tries:
            if num_tries > 0:
                print("EMPTY OUTPUT", num_tries)
            output = gpt(
                input_text,
                model_name=self.model_name,
                stop_sequences=self.stop_sequences,
                max_length=max_length,
                temperature=self.temperature,
            )
            num_tries += 1
        print(output)
        if self.answer_prefix is not None and self.answer_prefix not in output:
            input_text += " " + output + "\n" + self.answer_prefix
            output = gpt(
                input_text,
                model_name=self.model_name,
                stop_sequences=self.stop_sequences,
                max_length=max_length,
                temperature=self.temperature,
            )
        elif self.answer_prefix is not None:
            input_text += " " + output
            output = output.split(self.answer_prefix)[1].strip()
        """if self.answer_prefix is not None and self.answer_prefix in output:
            return output.split(self.answer_prefix)[1].strip(), input_text+" "+output"""
        return output, input_text + " " + output

    @classmethod
    def from_config(cls, cfg):
        model_name = cfg.get("model_name")
        num_tries = cfg.get("num_tries", 1)
        examples_path = cfg.get("examples_path", None)
        captions_per_image = cfg.get("captions_per_image", None)
        temperature = cfg.get("temperature", 0.0)
        context_separator = cfg.get("context_separator", " ")
        return cls(
            model_name,
            num_tries,
            examples_path,
            captions_per_image,
            temperature,
            context_separator,
        )
