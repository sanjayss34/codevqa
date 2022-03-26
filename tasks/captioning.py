import os
import json

import torch

from tasks.base_task import BaseTask
from common.registry import registry

import utils.blip_utils as utils


@registry.register_task('captioning')
class CaptionTask(BaseTask):
    ID_KEY = "image_id"
    CAP_KEY = "caption"

    def __init__(self, num_beams, max_len, min_len, evaluate):
        # TODO this has to be better defined by build_task(cfg) factory method and decouple Task from Config
        super().__init__()

        self.num_beams = num_beams
        self.max_len = max_len
        self.min_len = min_len
        self.evaluate = evaluate

    @classmethod
    def setup_task(cls, cfg):
        run_cfg = cfg.run_cfg

        num_beams = run_cfg.num_beams
        max_len = run_cfg.max_len
        min_len= run_cfg.min_len
        evaluate = run_cfg.evaluate

        return cls(
            num_beams=num_beams, 
            max_len=max_len, 
            min_len=min_len,
            evaluate=evaluate
        )


    def valid_step(self, model, samples):
        results = []

        # run_cfg = slf.cfg.run_cfg
        captions = model.generate(
                        samples, 
                        use_nucleus_sampling=False, 
                        num_beams=self.num_beams, 
                        max_length=self.max_len, 
                        min_length=self.min_len
                    )
        
        indices = samples[self.ID_KEY]
        for caption, index in zip(captions, indices):
            # results.append({"image_id": img_id.item(), "caption": caption})
            results.append({self.ID_KEY: index.item(), self.CAP_KEY: caption})

        return results
    
    def after_validation(self, val_result, split_name, epoch, **kwargs):
        val_result_file = utils.save_result(
            result=val_result, 
            result_dir=registry.get_path("result_dir"),
            filename='{}_epoch{}'.format(split_name, epoch), 
            remove_duplicate=self.ID_KEY
            )
        
        self._report_metrics(
            val_result_file=val_result_file,
            epoch=epoch,
            split_name=split_name
        )


    def _report_metrics(self, val_result_file, epoch, split_name):

        coco_gt_root = "annotation/coco_gt"

        if utils.is_main_process():
            # coco_val = utils.coco_caption_eval(self.config['coco_gt_root'], val_result_file, 'val')
            coco_val = utils.coco_caption_eval(coco_gt_root, val_result_file, split_name)

            if self.evaluate:
                log_stats = {**{f'val_{k}': v for k, v in coco_val.eval.items()}}

                with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                # save_obj = {
                #     'model': self.model_without_ddp.state_dict(),
                #     'optimizer': self.optimizer.state_dict(),
                #     'config': self.config,
                #     'epoch': epoch,
                # }

                # best ckpt
                # if coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4'] > best:
                #     best = coco_val.eval['CIDEr'] + coco_val.eval['Bleu_4']
                #     best_epoch = epoch
                #     torch.save(save_obj, os.path.join(self.output_dir, 'checkpoint_best.pth'))

                log_stats = {
                            **{f'val_{k}': v for k, v in coco_val.eval.items()},
                            'epoch': epoch,
                            # 'best_epoch': best_epoch,
                            }
                with open(os.path.join(self.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
