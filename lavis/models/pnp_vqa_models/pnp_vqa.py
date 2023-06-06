"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from lavis.common.registry import registry
from lavis.models.base_model import BaseModel
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import T5ForConditionalGeneration
from lavis.models.pnp_vqa_models import prepare_qa_input
from lavis.models.blip_models.blip_image_text_matching import compute_gradcam
from lavis.models import BlipVQA
from lavis.models.pnp_vqa_models.gpt_qa import gpt
from lavis.models.pnp_vqa_models.utils import anonymize_ast, collect_functions
from lavis.processors.blip_processors import BlipQuestionProcessor

import os
import json
from contextlib import nullcontext
import numpy as np
import time
from copy import deepcopy
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from collections import defaultdict, namedtuple
import ast
from copy import deepcopy
import random
from PIL import Image

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from typing import List


@registry.register_model("pnp_vqa")
class PNPVQA(BaseModel):
    """
    PNPVQA model consists of three submodels for zero-shot VQA:
        1. Image-questioning matching model
        2. Image captioning model
        3. Question answering model

    Supported model types:
        - base: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-base)
        - large: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-large)
        - 3b: BLIPITM, BLIPCaption, PNPUnifiedQAv2FiD (t5-3b)

    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("pnp_vqa", "base", is_eval=True)
        >>> model = load_model("pnp_vqa", "large", is_eval=True)
        >>> model = load_model("pnp_vqa", "3b", is_eval=True)
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "base": "configs/models/pnp-vqa/pnp_vqa_base.yaml",
        "large": "configs/models/pnp-vqa/pnp_vqa_large.yaml",
        "3b": "configs/models/pnp-vqa/pnp_vqa_3b.yaml",
        "gpt3": "configs/models/pnp-vqa/pnp_vqa_gpt3.yaml",
    }

    def __init__(
        self,
        image_question_matching_model,
        image_captioning_model,
        question_answering_model,
        simple_use_question_answering_model=False,
        image_selector_model=None,
        offload_model=False,
        subquestions=None,
        amp=False,
        vis_processor=None,
        examples=None,
        caption_full_image=False,
        captions_per_image=None,
        image_set=False,
        subquestion_example_selection=False,
        left_right_images=False,
        grounding_dino_path=None,
    ):
        super().__init__()

        self.image_question_matching_model = image_question_matching_model
        self.image_captioning_model = image_captioning_model
        self.question_answering_model = question_answering_model
        self.simple_use_question_answering_model = simple_use_question_answering_model
        self.vis_processor = vis_processor
        self.offload_model = offload_model
        self.subquestions_model_name = None
        self.subquestion_example_selection = subquestion_example_selection
        self.amp = amp
        self.caption_full_image = caption_full_image
        if grounding_dino_path:
            grounding_dino_config_path = os.path.join(grounding_dino_path, "config.py")
            grounding_dino_checkpoint_path = os.path.join(grounding_dino_path, "checkpoint.pth")
            grounding_dino_args = SLConfig.fromfile(grounding_dino_config_path)
            grounding_dino_args.device = self.image_question_matching_model.device
            self.grounding_dino_model = build_model(grounding_dino_args)
            checkpoint = torch.load(grounding_dino_checkpoint_path, map_location="cpu")
            load_res = self.grounding_dino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            print(load_res)
            self.grounding_dino_model.eval()
            self.grounding_dino_box_threshold = 0.3
        self.subquestions_template = None

        self.separate_visual_examples = False
        self.image_set = image_set
        if subquestions is not None:
            if subquestions["model_name"] != "none":
                self.subquestions_model_name = subquestions["model_name"]
            if "template" in subquestions and subquestions["template"] != "none":
                self.subquestions_template = subquestions[
                    "template"
                ]

            assert "prompt_path" in subquestions
            assert examples is not None
            with open(subquestions["prompt_path"]) as f:
                self.prompt = "".join(f.readlines()).strip()
            self.separate_visual_examples = True if "separate_visual_examples" in subquestions and subquestions["separate_visual_examples"] else False
            if hasattr(subquestions, "num_examples"):
                self.num_program_examples = subquestions.num_examples
            keys = None
            if examples["example_selection"] != "custom":
                assert "programs_path" in subquestions
                with open(subquestions["programs_path"]) as f:
                    program_dict = json.load(f)
                    program_dict = {int(key): program_dict[key] for key in program_dict}
                    self.program_dict = program_dict
                if "random" in examples["example_selection"]:
                    keys = random.sample(
                        list(program_dict.keys()), examples["num_examples"]
                    )
                if "embedding" not in examples["example_selection"]:
                    i = 1
                    if image_set:
                        for key in keys:
                            self.prompt += (
                                "\n# Image Set "
                                + str(i)
                                + ": "
                                + program_dict[key]["question"]
                                + '\nimages = open_images("ImageSet'
                                + str(i)
                                + '")\n'
                                + program_dict[key]["program"]
                            )
                            i += 1
                        self.prompt += "\n# Image Set " + str(i) + ": "
                    else:
                        for key in keys:
                            self.prompt += (
                                "\n# Image "
                                + str(i)
                                + ": "
                                + program_dict[key]["question"]
                                + '\nimg = open_image("Image'
                                + str(i)
                                + '.jpg")\n'
                                + program_dict[key]["program"]
                            )
                            i += 1
                        self.prompt += "\n# Image " + str(i) + ": "
            else:
                self.prompt += " "
            self.prompt_type = subquestions["prompt_path"].split(".")[-1]
            if "knowledge" in subquestions:
                with open(subquestions["knowledge"]["examples_path"]) as f:
                    self.knowledge_examples = json.load(f)
                self.knowledge_examples = {int(key): self.knowledge_examples[key] for key in self.knowledge_examples}
                if not isinstance(list(self.knowledge_examples.values())[0], list):
                    self.knowledge_examples = {key: [self.knowledge_examples[key]] for key in self.knowledge_examples}
                self.knowledge_prompt = (
                    "Answer each question.\n===\n"
                )
                if keys is None:
                    keys = list(self.program_dict.keys())
                for key in self.knowledge_examples:
                    if key in keys:
                        for ex in self.knowledge_examples[key]:
                            self.knowledge_prompt += "Q: " + ex["question"] + "\n"
                            self.knowledge_prompt += "A: " + ex["answer"] + "\n===\n"
                self.knowledge_prompt += "Q: |||\nA:"
                self.knowledge_model_name = subquestions["knowledge"]["model_name"]
        if examples is not None:
            with open(examples["captions_path"]) as f:
                captions = json.load(f)

            self.captions_dict = {datum["question_id"]: datum for datum in captions}
            if subquestions is not None:
                self.captions_dict = {key: self.captions_dict[key] for key in program_dict}
            self.example_selection = examples["example_selection"]
            self.num_examples = examples["num_examples"]
            if "examples_path" in examples:
                with open(examples["examples_path"]) as f:
                    self.visual_examples = json.load(f)
                    self.visual_examples = {datum["question_id"]: datum for datum in self.visual_examples}
            if subquestions is None or examples["example_selection"] == "custom":
                if "random" in examples["example_selection"]:
                    keys = random.sample(
                        list(self.captions_dict.keys()), examples["num_examples"]
                    )
                elif examples["example_selection"] == "custom":
                    with open(examples["examples_path"]) as f:
                        examples_info = json.load(f)
                    keys = [ex["question_id"] for ex in examples_info]
            if "embedding" in examples["example_selection"]:
                self.embedding_selection_model = SentenceTransformer(
                    "all-mpnet-base-v2"
                )
                self.embedding_selection_model.eval()
                self.embedding_selection_model = self.embedding_selection_model.to(
                    self.image_question_matching_model.device
                )
                keys = list(self.captions_dict.keys())
                self.example_questions = [
                    self.captions_dict[key]["question"] for key in keys
                ]
                if examples["example_selection"] == "embedding":
                    self.example_embeddings = self.embedding_selection_model.encode(
                        self.example_questions, convert_to_tensor=True
                    )
                else:
                    example_tokens = self.embedding_selection_tokenizer(self.example_questions, padding=True, truncation=True, return_tensors="pt").to(self.image_question_matching_model.device)
                    with torch.no_grad():
                        self.example_embeddings = self.embedding_selection_model(**example_tokens).pooler_output
                if subquestions is not None:
                    self.example_programs = [
                        program_dict[key]["program"] for key in keys
                    ]
                self.example_captions = [self.captions_dict[key] for key in keys]
            elif not hasattr(self.question_answering_model, "tokenizer") and not self.simple_use_question_answering_model:
                self.qa_prompt = self.question_answering_model.construct_prompt(
                    [self.captions_dict[key] for key in keys if key in self.captions_dict]
                )
                if examples["example_selection"] == "custom":
                    examples = [self.captions_dict[key] for key in keys]
                    for ex, ex_other in zip(examples, examples_info):
                        assert ex['question_id'] == ex_other['question_id']
                        ex['question'] = ex_other['question']
                        ex['answer'] = ex_other['answer']
                    self.qa_prompt2 = self.question_answering_model.construct_prompt(
                        examples
                    )
        if captions_per_image is not None:
            self.question_answering_model.captions_per_image = captions_per_image
        if left_right_images:
            self.question_answering_model.left_right_images = left_right_images

    def construct_prompts(self, questions, num_examples, keys_list=None):
        if "embedding" in self.example_selection:
            if self.example_selection == "embedding":
                question_embeddings = self.embedding_selection_model.encode(
                    questions, convert_to_tensor=True
                )
            else:
                question_tokens = self.embedding_selection_tokenizer(questions, padding=True, truncation=True, return_tensors="pt").to(self.embedding_selection_model.device)
                with torch.no_grad():
                    question_embeddings = self.embedding_selection_model(**question_tokens).pooler_output
            question_example_scores = question_embeddings @ self.example_embeddings.t().to(
                question_embeddings.device
            )
            example_indices = question_example_scores.argsort(dim=1, descending=True)
            # example_indices = example_indices[:,:num_examples].fliplr()
        input_texts = []
        qa_prompts = []
        if keys_list is None:
            keys_list = []
        for i in range(len(questions)):
            if "embedding" in self.example_selection:
                if len(keys_list) > i:
                    keys = keys_list[i]
                    indices = [[j for j in range(len(self.example_captions)) if self.example_captions[j]["question_id"] == key][0] for key in keys]
                else:
                    keys = [self.example_captions[example_indices[i,j].item()]["question_id"] for j in range(num_examples)]
                    indices = [example_indices[i,j].item() for j in range(num_examples)]
                if hasattr(self, "example_programs"):
                    input_text = (
                        self.prompt
                        + "\n"
                        + "\n".join(
                            [
                                ("# Image " if not self.image_set else "# Image Set ")
                                + str(j + 1)
                                + ": "
                                + self.example_questions[ind][
                                    0
                                ].upper()
                                + self.example_questions[ind][
                                    1:
                                ].lower()
                                + ('\nimg = open_image("Image' if not self.image_set else '\nimages = open_images("ImageSet')
                                + str(j + 1)
                                + '.jpg")\n'
                                + self.example_programs[ind]
                                for j, ind in enumerate(indices)
                            ]
                        )
                    )
                    input_text += (
                        ("\n# Image " if not self.image_set else "\n# Image Set ")
                        + str(num_examples + 1)
                        + ": "
                        + questions[i][0].upper()
                        + questions[i][1:].lower()
                    )
                    input_texts.append(input_text)
            elif self.example_selection == "random-each":
                keys = random.sample(self.program_dict.keys(), num_examples)
                input_text = (
                    self.prompt.split("\n# Image 1")[0]
                    + "\n"
                    + "\n".join(
                        [
                            ("# Image " if not self.image_set else "# Image Set ")
                            + str(j + 1)
                            + ": "
                            + self.program_dict[key]['question'][0].upper()
                            + self.program_dict[key]['question'][1:].lower()
                            + ('\nimg = open_image("Image' if not self.image_set else '\nimages = open_images("ImageSet')
                            + str(j+1)
                            + '.jpg")\n'
                            + self.program_dict[key]['program']
                            for j, key in enumerate(keys)
                        ]
                    )
                )
                input_text += (
                    ("\n# Image " if not self.image_set else "\n# Image Set ")
                    + str(num_examples + 1)
                    + ": "
                    + questions[i][0].upper()
                    + questions[i][1:].lower()
                )
                keys_list.append(keys)
                input_texts.append(input_text)
            if "embedding" in self.example_selection:
                if hasattr(self.question_answering_model, "construct_prompt"):
                    qa_prompts.append(
                        self.question_answering_model.construct_prompt(
                            [
                                self.example_captions[j]
                                for j in indices
                            ]
                        )
                    )
                else:
                    qa_prompts.append("")
                keys_list.append(
                    keys
                )
        return input_texts, qa_prompts, keys_list

    def forward_itm(self, samples, block_num=7, token_index=1):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
        """
        # print('itm', samples['image'].shape)
        image = samples["image"]
        question = [text.strip("?") for text in samples["text_input"]]
        tokenized_text = self.image_question_matching_model.tokenizer(
            question,
            padding="longest",
            truncation=True,
            max_length=self.image_question_matching_model.max_txt_len,
            return_tensors="pt",
        ).to(self.image_question_matching_model.device)
        with torch.set_grad_enabled(True):
            gradcams, _ = compute_gradcam(
                model=self.image_question_matching_model,
                visual_input=image,
                text_input=question,
                tokenized_text=tokenized_text,
                block_num=block_num,
            )

        gradcams = [gradcam_[token_index] for gradcam_ in gradcams]
        samples["gradcams"] = torch.stack(gradcams).view(samples["image"].size(0), -1)
        dim = int(np.sqrt(samples["gradcams"].shape[1]))
        samples["gradcams"] = samples["gradcams"].view(-1, dim, dim)
        if "region" in samples:
            region = samples["region"]
            for i, box in enumerate(region):
                samples["gradcams"][i, :, : box[0]] = 0
                samples["gradcams"][i, : box[1], :] = 0
                samples["gradcams"][i, :, box[2] :] = 0
                samples["gradcams"][i, box[3] :, :] = 0
        samples["gradcams"] = samples["gradcams"].reshape(
            samples["image"].size(0), -1
        )
        return samples

    def forward_cap(
        self,
        samples,
        cap_max_length=20,
        cap_min_length=0,
        top_p=1,
        top_k=50,
        repetition_penalty=1.0,
        num_captions=100,
        num_patches=20,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions generated for each image.
            num_patches (int): Number of patches sampled for each image.

        Returns:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
                - captions (nested list): A nested list of strings of total length batch_size * num_captions
        """
        # print('cap', samples['image'].shape)
        encoder_out = self.image_captioning_model.forward_encoder(samples)
        captions = [[] for _ in range(encoder_out.size(0))]
        gen_prompts = [[] for _ in range(encoder_out.size(0))]

        min_num_captions = 0

        while min_num_captions < num_captions:
            encoder_out_samples = []
            for i in range(num_captions):
                if self.caption_full_image:
                    patch_id = (
                        torch.arange(samples["gradcams"].shape[1])
                        .unsqueeze(0)
                        .repeat(encoder_out.size(0), 1)
                        .to(encoder_out.device)
                        + 1
                    )
                else:
                    patch_id = (
                        torch.multinomial(
                            samples["gradcams"].to(self.image_captioning_model.device),
                            num_patches,
                        ).reshape(encoder_out.size(0), -1)
                        + 1
                    )
                patch_id = (
                    patch_id.sort(dim=1)
                    .values.unsqueeze(-1)
                    .expand(-1, -1, encoder_out.size(2))
                )
                encoder_out_sample = torch.gather(encoder_out, 1, patch_id)
                encoder_out_samples.append(encoder_out_sample)

            stacked = torch.stack(encoder_out_samples, dim=1)
            image_embeds = torch.flatten(
                stacked, start_dim=0, end_dim=1
            )  # (bsz*num_seq, num_patch, dim)

            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
                self.image_captioning_model.device
            )
            model_kwargs = {
                "encoder_hidden_states": image_embeds,
                "encoder_attention_mask": image_atts,
            }

            prompt = [self.image_captioning_model.prompt] * image_embeds.size(0)

            prompt = self.image_captioning_model.tokenizer(
                prompt, return_tensors="pt"
            ).to(self.image_captioning_model.device)
            prompt.input_ids[:, 0] = self.image_captioning_model.tokenizer.bos_token_id
            prompt.input_ids = prompt.input_ids[:, :-1]

            decoder_out = self.image_captioning_model.text_decoder.generate(
                input_ids=prompt.input_ids,
                max_length=cap_max_length,
                min_length=cap_min_length,
                do_sample=True,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
                eos_token_id=self.image_captioning_model.tokenizer.sep_token_id,
                pad_token_id=self.image_captioning_model.tokenizer.pad_token_id,
                repetition_penalty=repetition_penalty,
                **model_kwargs
            )

            outputs = self.image_captioning_model.tokenizer.batch_decode(
                decoder_out, skip_special_tokens=True
            )

            for counter, output in enumerate(outputs):
                ind = counter // num_captions
                if len(captions[ind]) < num_captions:
                    caption = output[len(self.image_captioning_model.prompt) :]
                    overlap_caption = [1 for caps in captions[ind] if caption in caps]
                    if len(overlap_caption) == 0:
                        captions[ind].append(caption)

            min_num_captions = min([len(i) for i in captions])

        samples["captions"] = captions
        return samples

    def forward_qa(
        self,
        samples,
        num_beams=1,
        max_len=20,
        min_len=0,
        internal_bsz_fid=1,
        num_captions=100,
        num_captions_fid=1,
        subquestion=False,
        keys_list=None,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
                - text_input (list): A list of strings of length batch_size
                - gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
                - captions (nested list): A nested list of strings of total length batch_size * num_captions
                - question_captions (nested list): A nested list of concatenated strings of questions and captions
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            internal_bsz_fid (int): Internal batch size when using FiD decoding.
            num_captions (int): Number of captions generated for each image.
            num_captions_fid (int): Number of captions concatenated with a question during FiD decoding.

        Returns:
            List: A list of strings, each string is an answer.
        """
        # print('qa', samples['image'].shape)
        if isinstance(self.question_answering_model, BlipVQA):
            return self.question_answering_model.predict_answers(
                samples, max_len=max_len, min_len=min_len, inference_method="generate"
            )
        pred_answers = []
        if hasattr(self.question_answering_model, "tokenizer"):
            prepare_qa_input(
                samples, num_captions=num_captions, num_captions_fid=num_captions_fid
            )

            question_captions = samples["question_captions"]
            question_captions_chunk = [
                question_captions[i : i + internal_bsz_fid]
                for i in range(0, len(question_captions), internal_bsz_fid)
            ]
            question_captions_chunk = list(chain(*question_captions_chunk))
        else:
            question_captions_chunk = samples["captions"]

        if hasattr(self, "captions_dict"):
            examples_dict = self.captions_dict
        if self.separate_visual_examples and subquestion:
            examples_dict = self.visual_examples
        if hasattr(self, "example_embeddings"):
            if keys_list is None:
                _, qa_prompts, _ = self.construct_prompts(samples["text_input"], num_examples=self.num_examples)
            else:
                keys_list = [keys if len(keys) == self.num_examples else self.construct_prompts(samples["text_input"][i:i+1], num_examples=self.num_examples)[2][0] for i, keys in enumerate(keys_list)]
                keys = [[key for key in keys if key in examples_dict] for keys in keys_list]
                qa_prompts = [
                    self.question_answering_model.construct_prompt(
                        [
                            examples_dict[key]
                            for key in keys_list[index]
                        ]
                    )
                    for index in range(len(samples["text_input"]))
                ]
        elif (
            hasattr(self, "example_selection")
            and self.example_selection == "random-each"
        ):
            all_keys = list(examples_dict.keys())
            if keys_list is None:
                keys_list = [random.sample(all_keys, self.num_examples) for _ in samples["text_input"]]
            keys_list = [keys if len(keys) == self.num_examples else random.sample(all_keys, self.num_examples) for keys in keys_list]
            keys = [[key for key in keys if key in examples_dict] for keys in keys_list]
            qa_prompts = [
                self.question_answering_model.construct_prompt(
                    [
                        examples_dict[key]
                        for key in keys_list[index]
                    ]
                )
                for index in range(len(samples["text_input"]))
            ]
        elif hasattr(self, "qa_prompt"):
            qa_prompts = [self.qa_prompt for _ in samples["text_input"]]
        else:
            qa_prompts = ["" for _ in samples["text_input"]]

        index = 0
        for question_caption, question, captions, qa_prompt in zip(
            question_captions_chunk,
            samples["text_input"],
            samples["captions"],
            qa_prompts,
        ):
            if hasattr(self.question_answering_model, "tokenizer"):
                question_caption_input = self.question_answering_model.tokenizer(
                    question_caption,
                    padding="longest",
                    truncation=True,
                    return_tensors="pt",
                ).to(self.question_answering_model.device)

                question_caption_input.input_ids = (
                    question_caption_input.input_ids.reshape(
                        internal_bsz_fid, -1, question_caption_input.input_ids.size(1)
                    )
                )
                question_caption_input.attention_mask = (
                    question_caption_input.attention_mask.reshape(
                        internal_bsz_fid,
                        -1,
                        question_caption_input.attention_mask.size(1),
                    )
                )

                outputs = self.question_answering_model.generate(
                    input_ids=question_caption_input.input_ids,
                    attention_mask=question_caption_input.attention_mask,
                    num_beams=num_beams,
                    min_length=min_len,
                    max_length=max_len,
                )
                for output in outputs:
                    pred_answer = self.question_answering_model.tokenizer.decode(
                        output, skip_special_tokens=True
                    )
                    pred_answers.append(pred_answer)
            else:
                answer, full_answer = self.question_answering_model.generate(
                    question, captions, qa_prompt
                )
                pred_answers.append(answer)
                if full_answer is not None:
                    samples["captions"][index].append(full_answer)
            index += 1

        return pred_answers

    def simple_predict_answers(
        self,
        samples,
        num_beams=1,
        inference_method="generate",
        max_len=20,
        min_len=0,
        internal_bsz_fid=1,
        num_captions=50,
        num_captions_qa=50,
        num_captions_fid=1,
        cap_max_length=20,
        cap_min_length=10,
        top_k=50,
        top_p=1,
        repetition_penalty=1,
        num_patches=50,
        block_num=7,
        subquestion=False,
        keys_list=None,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W). Default H=480, W=480.
                - text_input (str or [str]): String or a list of strings, each string is a question.
                                             The number of questions must be equal to the batch size. If a single string, will be converted to a list of string, with length 1 first.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            inference_method (str): Inference method. Must be "generate". The model will generate answers.
            max_len (int): Maximum length of generated answers.
            min_len (int): Minimum length of generated answers.
            internal_bsz_fid (int): Internal batch size when using FiD decoding.
            num_captions (int): Number of captions generated for each image.
            num_captions_fid (int): Number of captions concatenated with a question during FiD decoding.
            cap_max_length (int): The maximum length of the caption to be generated.
            cap_min_length (int): The minimum length of the caption to be generated.
            top_k (float): The number of the highest probability tokens for top-k sampling.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_patches (int): Number of patches sampled for each image.
            block_num (int): The index of cross-attention block for gradcam computation.

        Returns:
            List: A list of strings, each string is an answer.
            gradcams (torch.Tensor): A tensor of shape (batch_size, H*W)
            captions (nested list): A nested list of strings of total length batch_size * num_captions
        """

        assert inference_method in [
            "generate",
        ], "Inference method must be 'generate', got {}.".format(inference_method)

        if isinstance(samples["text_input"], str):
            samples["text_input"] = [samples["text_input"]]

        if "image" in samples:
            visual_batch_size = samples["image"].size(0)
        elif "images" in samples:
            visual_batch_size = samples["images"].size(0)
        assert (
            len(samples["text_input"]) == visual_batch_size
        ), "The number of questions must be equal to the batch size."

        if self.offload_model:
            self.question_answering_model.to("cpu")
            
        if self.simple_use_question_answering_model:
            answers = []
            for i in range(visual_batch_size):
                qa_samples = {"prompt": "Question: " + samples["text_input"][i] + " Answer:", "image": samples["image"][i].unsqueeze(0).to(device=self.question_answering_model.device)}
                # qa_samples = {"image": samples["image"][i].unsqueeze(0).to(device=self.question_answering_model.device)}
                answers.append(self.question_answering_model.generate(qa_samples)[0])
            return answers, [None]*len(answers), [None]*len(answers)
        
        if "image" in samples:
            samples = self.forward_itm(samples, block_num=block_num)

            samples = self.forward_cap(
                samples,
                cap_max_length=cap_max_length,
                cap_min_length=cap_min_length,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_captions=num_captions,
                num_patches=num_patches,
            )
        elif "video" in samples:
            samples = self.forward_frames(samples)
            samples = self.forward_video_cap(
                samples,
                cap_max_length=cap_max_length,
                cap_min_length=cap_min_length,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_captions=num_captions,
                num_patches=num_patches,
                block_num=block_num,
            )
        elif "images" in samples:
            samples["captions"] = {}
            for i in range(samples["images"].size(1)):
                samples_i = deepcopy(samples)
                samples_i["image"] = samples["images"][:, i, :, :, :]
                samples_i = self.forward_itm(samples_i, block_num=block_num)
                samples_i = self.forward_cap(
                    samples_i,
                    cap_max_length=cap_max_length,
                    cap_min_length=cap_min_length,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    num_captions=num_captions,
                    num_patches=num_patches,
                )
                samples["captions"][i] = samples_i["captions"]
            combined_captions = [[""] for _ in range(visual_batch_size)]
            separate_captions = [[] for _ in range(visual_batch_size)]
            for i in samples["captions"]:
                for j in range(visual_batch_size):
                    if samples["num_images"][j].item() > i:
                        combined_captions[j][0] += (
                            "\nImage "
                            + str(i + 1)
                            + ": "
                            + ". ".join(samples["captions"][i][j])
                        )
                        separate_captions[j].append(samples["captions"][i][j])
            combined_captions = [[caption[0].strip()] for caption in combined_captions]
            if hasattr(self.question_answering_model, "tokenizer"):
                samples["captions"] = combined_captions
            else:
                samples["captions"] = separate_captions
            samples["image"] = samples["images"][:, 0]
            samples["gradcams"] = samples_i["gradcams"]

        if self.offload_model:
            self.image_question_matching_model.to("cpu")
            self.image_captioning_model.to("cpu")
        self.question_answering_model.to(samples["image"].device)
        if self.offload_model:
            samples["image"] = samples["image"].to("cpu")
        torch.cuda.empty_cache()

        pred_answers = self.forward_qa(
            samples,
            num_beams=num_beams,
            max_len=max_len,
            min_len=min_len,
            internal_bsz_fid=internal_bsz_fid,
            num_captions=num_captions_qa,
            num_captions_fid=num_captions_fid,
            subquestion=subquestion,
            keys_list=keys_list,
        )

        if self.offload_model:
            self.image_question_matching_model.to(self.question_answering_model.device)
            self.image_captioning_model.to(self.question_answering_model.device)
        if "images" in samples:
            samples["captions"] = separate_captions

        return pred_answers, samples["captions"], samples["gradcams"]

    def argmax_text_location(self, samples, token_index):
        if isinstance(token_index, int):
            token_index = [token_index]
        gradcams = []
        dim = 24
        for t in token_index:
            samples_w_gradcam = self.forward_itm(samples=samples, token_index=t)
            gradcam = samples_w_gradcam["gradcams"].reshape(dim, dim)
            gradcams.append(gradcam)
        gradcam = torch.stack(gradcams).mean(0)
        argmax_index = gradcam.argmax().item()
        yloc = argmax_index // gradcam.shape[1]
        xloc = argmax_index % gradcam.shape[1]
        return (xloc, dim - yloc), gradcam

    def get_image_portion(self, raw_image, x0, y0, x1, y1):
        dim = 24
        y0 = int((dim - y0) * raw_image.height / dim)
        y1 = int((dim - y1) * raw_image.height / dim)
        x0 = int(x0 * raw_image.width / dim)
        x1 = int(x1 * raw_image.width / dim)
        mask = Image.new("L", raw_image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([(x0, y0), (x1, y1)], fill=255)
        blurred = raw_image.filter(ImageFilter.GaussianBlur(52))
        blurred.paste(raw_image, mask=mask)
        return blurred

    def predict_answers(
        self,
        samples,
        num_beams=1,
        inference_method="generate",
        max_len=20,
        min_len=0,
        internal_bsz_fid=1,
        num_captions=50,
        num_captions_qa=50,
        num_captions_fid=1,
        cap_max_length=20,
        cap_min_length=10,
        top_k=50,
        top_p=1,
        repetition_penalty=1,
        num_patches=50,
        block_num=7,
    ):
        with torch.cuda.amp.autocast() if self.amp else nullcontext():
            if self.subquestions_model_name is None:
                return self.simple_predict_answers(
                    samples,
                    num_beams,
                    inference_method,
                    max_len,
                    min_len,
                    internal_bsz_fid,
                    num_captions,
                    num_captions_qa,
                    num_captions_fid,
                    cap_max_length,
                    cap_min_length,
                    top_k,
                    top_p,
                    repetition_penalty,
                    num_patches,
                    block_num,
                )
            pred_answers = []
            pred_captions = []
            pred_gradcams = []
            txt_processor = BlipQuestionProcessor()
            input_texts = None
            keys_list = None
            error_count = 0
            if hasattr(self, "example_embeddings") or "-each" in self.example_selection:
                if hasattr(self, "num_program_examples"):
                    num_examples = self.num_program_examples
                else:
                    num_examples = self.num_examples
                input_texts, _, keys_list = self.construct_prompts(samples["text_input"], num_examples=num_examples)
            for index in range(len(samples["text_input"])):
                def simple_answer(q_text, region=None, interval=None, subquestion=False, pass_keys=True):
                    q_samples = {"text_input": [txt_processor(q_text)]}
                    if "image" in samples:
                        q_samples["image"] = samples["image"][index : index + 1]
                    elif "images" in samples:
                        q_samples["images"] = samples["images"][index : index + 1]
                        q_samples["num_images"] = samples["num_images"][
                            index : index + 1
                        ]
                    if region is not None:
                        assert "image" in q_samples
                        region = [region[0], 24 - region[3], region[2], 24 - region[1]]
                        q_samples["region"] = [region]
                    return self.simple_predict_answers(
                        q_samples,
                        num_beams,
                        inference_method,
                        max_len,
                        min_len,
                        internal_bsz_fid,
                        num_captions,
                        num_captions_qa,
                        num_captions_fid,
                        cap_max_length,
                        cap_min_length,
                        top_k,
                        top_p,
                        repetition_penalty,
                        num_patches,
                        block_num,
                        subquestion,
                        keys_list[index:index+1] if (keys_list is not None and (pass_keys or not self.subquestion_example_selection)) else None
                    )

                assert self.prompt_type == "py"
                if input_texts is not None:
                    input_text = input_texts[index]
                else:
                    input_text = (
                        self.prompt.strip()
                        + " "
                        + samples["text_input"][index][0].upper()
                        + samples["text_input"][index][1:].lower()
                    )
                print(samples.keys())
                instance_captions = []
                instance_gradcams = []

                visual_subquestion = {"value": False}

                raw_image = None
                if "image_path" in samples:
                    raw_image = Image.open(samples["image_path"][index]).convert("RGB")
                if "image_paths" in samples:
                    samples["image_paths"][index] = samples["image_paths"][index].split("|||")

                class Box(namedtuple('Box', 'min_x min_y max_x max_y')):
                    pass

                def knowledge_query(q):
                    knowledge_prompt = self.knowledge_prompt
                    if "-each" in self.example_selection or "embedding" == self.example_selection:
                        knowledge_prompt = self.knowledge_prompt.split('\n')[0]+"\n===\n"
                        for key in self.knowledge_examples.keys():
                            if key in keys_list[index]:
                                for ex in self.knowledge_examples[key]:
                                    knowledge_prompt += "Q: " + ex["question"] + "\n"
                                    knowledge_prompt += "A: " + ex["answer"] + "\n===\n"
                        knowledge_prompt += "Q: |||\nA:"
                    answer = gpt(
                        knowledge_prompt.replace("|||", q),
                        model_name=self.knowledge_model_name,
                        stop_sequences=["\n", "\r"],
                        temperature=0,
                        max_length=10,
                        logit_bias={1906: -100, 7200: -100, 284: -100, 532: -100, 12: -100, 11: -100, 14: -100},
                    ).lower()
                    instance_captions.append(
                        knowledge_prompt.replace("|||", q) + " " + answer
                    )
                    print(answer)
                    return answer

                def visual_query(img, q):
                    if isinstance(img, int):
                        samples["image"] = samples["images"][index, img : img + 1]
                    print("visual subquestion", visual_subquestion["value"])
                    ans, caps, gcams = simple_answer(q, subquestion=visual_subquestion["value"], pass_keys=("embedding" not in self.example_selection))
                    if isinstance(img, int):
                        instance_captions.append([img]+[caps])
                    else:
                        instance_captions.append(caps)
                    answer = ans[0].lower()
                    if isinstance(img, int):
                        del samples["image"]
                    return answer

                query = visual_query

                def get_pos(img, refexp, *args):
                    if isinstance(img, str):
                        refexp = img
                        img = None
                    full_tokens = self.image_question_matching_model.tokenizer(
                        samples["text_input"][index]
                    )["input_ids"]
                    refexp_tokens = self.image_question_matching_model.tokenizer(
                        refexp
                    )["input_ids"][1:-1]
                    token_indices = None
                    for i in range(len(full_tokens)):
                        if full_tokens[i : i + len(refexp_tokens)] == refexp_tokens:
                            token_indices = [
                                1 + j for j in range(i, i + len(refexp_tokens))
                            ]
                    if token_indices is None:
                        token_indices = []
                        for i in range(len(full_tokens)):
                            if full_tokens[i] in refexp_tokens:
                                token_indices.append(1 + i)
                        if len(token_indices) == 0:
                            token_indices = [1]
                    pos, gradcam = self.argmax_text_location(
                        {
                            "image": samples["image"][index : index + 1] if "image" in samples else samples["images"][index][img:img+1],
                            "text_input": [samples["text_input"][index]],
                        },
                        token_indices,
                    )
                    instance_gradcams.append(gradcam)
                    return pos

                def find_object(img, refexp):
                    grounding_dino_transform = T.Compose(
                        [
                            T.RandomResize([800], max_size=1333),
                            T.ToTensor(),
                            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ]
                    )
                    if isinstance(img, int):
                        print(len(samples['image_paths'][index]), img)
                        raw_image = Image.open(samples["image_paths"][index][img]).convert("RGB")
                    else:
                        raw_image = Image.open(samples["image_path"][index]).convert("RGB")
                    image_t, _ = grounding_dino_transform(raw_image, None)
                    text = refexp.lower()
                    text = text.strip()
                    if not text.endswith("."):
                        text = text + "."
                    self.grounding_dino_model = self.grounding_dino_model.to(self.image_question_matching_model.device)
                    image_t = image_t.to(self.image_question_matching_model.device)
                    with torch.no_grad():
                        outputs = self.grounding_dino_model(image_t[None], captions=[text])
                    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
                    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

                    # filter output
                    logits_filt = logits.clone()
                    boxes_filt = boxes.clone()
                    filt_mask = logits_filt.max(dim=1)[0] > self.grounding_dino_box_threshold
                    logits_filt = logits_filt[filt_mask]  # num_filt, 256
                    boxes_filt = boxes_filt[filt_mask]
                    return [Box(*box) for box in boxes_filt]

                def find_matching_image(images, refexp):
                    prompt = "a picture of "
                    scores = self.image_question_matching_model(
                        {
                            "image": samples["images"][index, images],
                            "text_input": [prompt+refexp],
                        },
                        match_head="itc",
                    ).view(-1)
                    instance_captions.append(['find_matching_image', refexp, str(scores.tolist())])
                    return images[scores.argmax().item()]

                def open_image(filename):
                    return raw_image

                def open_images(path):
                    return list(range(samples["num_images"][index].item()))

                output = gpt(
                    input_text,
                    model_name=self.subquestions_model_name,
                    stop_sequences=["\n# Image", "\r# Image", "\n# Video", "\r# Video"],
                    temperature=0,
                    max_length=512,
                )
                post_gpt_time = time.time()
                print(input_text)
                print(output)
                output = output.replace("\r", "\n")
                code = (
                    self.prompt.split("from utils")[0]
                    + input_text.split("\n")[-1]
                    + "\nprint(dir())\n"
                    + output
                )
                if len([line for line in code.split('\n') if '_query' in line]) > 1:
                    visual_subquestion["value"] = True
                answer = "no"
                locals_dict = {
                    "query": query,
                    "get_pos": get_pos,
                    "open_image": open_image,
                    "visual_query": visual_query,
                    "knowledge_query": knowledge_query,
                    "find_matching_image": find_matching_image,
                    "open_images": open_images,
                    "find_object": find_object,
                    "Box": Box,
                }
                finished = False
                try:
                    exec(code, globals(), locals_dict)
                    if "answer" in locals_dict:
                        answer = str(locals_dict["answer"])
                    finished = True
                except Exception as e:
                    print("ERROR:", e)
                    print(code)
                    error_count += 1
                    if "images" in samples:
                        if "image" in samples:
                            del samples["image"]
                if not finished:
                    try:
                        ans, caps, gcams = simple_answer(samples["text_input"][index])
                        instance_captions.append(caps[0])
                        instance_gradcams.append(gcams[0])
                        answer = ans[0].lower()
                    except Exception as e:
                        instance_captions.append("error: "+str(e))
                        instance_gradcams.append(np.ones((24, 24)))
                        pass
                print("ANSWER:", answer)
                pred_answers.append(answer)
                pred_captions.append(instance_captions + [input_text + "\n" + output])
                pred_gradcams.append(instance_gradcams)
                elapsed = time.time() - post_gpt_time
                if elapsed < 3:
                    time.sleep(
                        3 - elapsed
                    )  # To avoid a rate limit error from OpenAI with Codex
            print("ERROR COUNT: ", error_count)
        return pred_answers, pred_captions, pred_gradcams

    @classmethod
    def from_config(cls, model_config):
        print(model_config)
        itm_config = model_config.image_question_matching_model
        cap_config = model_config.image_captioning_model
        qa_config = model_config.question_answering_model
        print(qa_config)

        itm_cls = registry.get_model_class(itm_config.arch)
        cap_cls = registry.get_model_class(cap_config.arch)
        qa_cls = registry.get_model_class(qa_config.arch)

        image_question_matching_model = itm_cls.from_config(itm_config)
        image_captioning_model = cap_cls.from_config(cap_config)
        question_answering_model = qa_cls.from_config(qa_config)

        vis_processor_cfg = model_config.get("vis_processor", None)
        vis_processor = None
        if vis_processor_cfg is not None:
            vis_processor = registry.get_processor_class(
                vis_processor_cfg.name
            ).from_config(vis_processor_cfg)
        simple_use_question_answering_model = model_config.get("simple_use_question_answering_model", False)
        caption_full_image = model_config.get("caption_full_image", False)
        captions_per_image = model_config.get("captions_per_image", None)
        image_set = model_config.get("image_set", None)
        subquestion_example_selection = model_config.get("subquestion_example_selection", False)
        left_right_images = model_config.get("left_right_images", False)

        grounding_dino_path = model_config.get("grounding_dino_path", None)

        model = cls(
            image_question_matching_model=image_question_matching_model,
            image_captioning_model=image_captioning_model,
            question_answering_model=question_answering_model,
            simple_use_question_answering_model=simple_use_question_answering_model,
            image_selector_model=None,
            offload_model=True if model_config.model_type == "3b" else False,
            subquestions=model_config.subquestions
            if hasattr(model_config, "subquestions")
            else None,
            examples=model_config.examples
            if hasattr(model_config, "examples")
            else None,
            amp=model_config.get("amp", False),
            vis_processor=vis_processor,
            caption_full_image=caption_full_image,
            captions_per_image=captions_per_image,
            image_set=image_set,
            subquestion_example_selection=subquestion_example_selection,
            left_right_images=left_right_images,
            grounding_dino_path=grounding_dino_path
        )

        return model
