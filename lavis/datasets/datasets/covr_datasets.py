"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random

from PIL import Image
import torch

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from lavis.datasets.datasets.gqa_datasets import __DisplMixin

from collections import OrderedDict

class COVRDataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def shuffle(self):
        random.shuffle(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        images = []
        for scene_id in ann["scenes"]:
            if scene_id.isnumeric():
                image_path = os.path.join(self.vis_root, "VG_100K", scene_id+".jpg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.vis_root, "VG_100K_2", scene_id+".jpg")
            elif scene_id.count('-') == 3:
                # NLVR2
                image_path = os.path.join(self.vis_root, scene_id+".png")
            else:
                image_path = os.path.join(self.vis_root, "of500_images", scene_id.split("_")[0], scene_id+".jpg")
            image = Image.open(image_path).convert("RGB")
            images.append(self.vis_processor(image))

        images = torch.stack(images)
        question = self.text_processor(ann["question"])

        if isinstance(ann["answer"], bool) and ann["answer"] == True:
            ann["answer"] = "yes"
        elif isinstance(ann["answer"], bool) and ann["answer"] == False:
            ann["answer"] = "no"
        answers = [str(ann["answer"])]
        weights = [1]

        return {
            "images": images,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


class COVREvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. gqa/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        print(ann_paths)
        self.annotation = json.load(open(ann_paths[0]))

        ## TODO: support inference method == 'ranking'
        answer_list_path = ann_paths[1] if len(ann_paths) > 1 else ''
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()
        print('DATASET LENGTH', len(self))

    def shuffle(self):
        random.shuffle(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        images = []
        image_paths = []
        for scene_id in ann["scenes"]:
            if scene_id.isnumeric():
                image_path = os.path.join(self.vis_root, "VG_100K", scene_id+".jpg")
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.vis_root, "VG_100K_2", scene_id+".jpg")
            elif scene_id.count('-') == 3:
                # NLVR2
                image_path = os.path.join(self.vis_root, scene_id+".png")
            else:
                image_path = os.path.join(self.vis_root, "of500_images", scene_id.split("_")[0], scene_id+".jpg")
            image = Image.open(image_path).convert("RGB")
            images.append(self.vis_processor(image))
            image_paths.append(image_path)

        images = torch.stack(images)
        question = self.text_processor(ann["question"])
        if question[-1] != "?":
            question = "Is it true that "+question.lower()+"?"

        if "answer" in ann:
            # answer is a string
            if isinstance(ann["answer"], bool) and ann["answer"] == True:
                ann["answer"] = "yes"
            elif isinstance(ann["answer"], bool) and ann["answer"] == False:
                ann["answer"] = "no"
            answer = str(ann["answer"])
        else:
            answer = None

        image_paths = "|||".join(image_paths)
        return {
            "images": images,
            "image_paths": image_paths,
            "text_input": question,
            "answer": answer,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "num_images": images.size(0)
        }
