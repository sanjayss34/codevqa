"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.aok_vqa_datasets import AOKVQADataset, AOKVQAEvalDataset
from lavis.datasets.datasets.coco_vqa_datasets import COCOVQADataset, COCOVQAEvalDataset
from lavis.datasets.datasets.vg_vqa_datasets import VGVQADataset
from lavis.datasets.datasets.gqa_datasets import GQADataset, GQAEvalDataset
from lavis.datasets.datasets.covr_datasets import COVRDataset, COVREvalDataset

class VQABuilder(BaseDatasetBuilder):
    train_dataset_cls = BaseDataset
    eval_dataset_cls = BaseDataset

    def build(self):
        datasets = super().build()

        for split in datasets:
            shuffle = self.config.get("shuffle", False)
            if shuffle:
                datasets[split].shuffle()
            start = self.config.get("start", None)
            if start is not None:
                datasets[split].annotation = datasets[split].annotation[start:]
            limit = self.config.get("limit", None)
            if limit is not None:
                datasets[split].annotation = datasets[split].annotation[:limit]
        return datasets

@registry.register_builder("coco_vqa")
class COCOVQABuilder(VQABuilder):
    train_dataset_cls = COCOVQADataset
    #eval_dataset_cls = COCOVQADataset
    eval_dataset_cls = COCOVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/defaults_vqa.yaml",
        "eval": "configs/datasets/coco/eval_vqa.yaml",
        "train100": "configs/datasets/coco/eval_vqa_train100.yaml",
        "train200": "configs/datasets/coco/eval_vqa_train200.yaml",
        "train2000": "configs/datasets/coco/eval_vqa_train2000.yaml",
        "val4000": "configs/datasets/coco/eval_vqa_val4000.yaml"
    }


@registry.register_builder("vg_vqa")
class VGVQABuilder(COCOVQABuilder):
    train_dataset_cls = VGVQADataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/vg/defaults_vqa.yaml"}


@registry.register_builder("ok_vqa")
class OKVQABuilder(COCOVQABuilder):
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/okvqa/defaults.yaml",
        "sample200": "configs/datasets/okvqa/sample200.yaml",
        "sample1000": "configs/datasets/okvqa/sample1000.yaml",
        "train100": "configs/datasets/okvqa/train100.yaml"
    }

@registry.register_builder("aok_vqa")
class AOKVQABuilder(VQABuilder):
    train_dataset_cls = AOKVQADataset
    eval_dataset_cls = AOKVQAEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/aokvqa/defaults.yaml"}


@registry.register_builder("gqa")
class GQABuilder(VQABuilder):
    def build(self):
        datasets = super().build()
        if self.config.get("shuffle", False):
            for split in datasets:
                datasets[split].shuffle()
        return datasets
    train_dataset_cls = GQADataset
    eval_dataset_cls = GQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/gqa/defaults.yaml",
        "balanced_val": "configs/datasets/gqa/balanced_val.yaml",
        "sample200": "configs/datasets/gqa/sample200.yaml",
        "train100": "configs/datasets/gqa/train100.yaml",
        "balanced_testdev": "configs/datasets/gqa/balanced_testdev.yaml",
        "sample2000": "configs/datasets/gqa/sample2000.yaml",
    }

@registry.register_builder("covr")
class COVRBuilder(VQABuilder):
    train_dataset_cls = COVRDataset
    eval_dataset_cls = COVREvalDataset

    DATASET_CONFIG_DICT = {
        "train100": "configs/datasets/covr/train100.yaml",
        "sample200": "configs/datasets/covr/sample200.yaml",
        "sample1000": "configs/datasets/covr/sample1000.yaml",
        "val": "configs/datasets/covr/val.yaml",
        "val_paraphrased": "configs/datasets/covr/val_paraphrased.yaml",
        "test": "configs/datasets/covr/test.yaml",
        "test_paraphrased": "configs/datasets/covr/test_paraphrased.yaml",
        "nlvr2_incontext50": "configs/datasets/covr/nlvr2_incontext50.yaml",
        "nlvr2_train2000": "configs/datasets/covr/nlvr2_train2000.yaml",
        "nlvr2_test": "configs/datasets/covr/nlvr2_test.yaml"
    }
