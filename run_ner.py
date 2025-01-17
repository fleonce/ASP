import json
import os
import sys
import logging
import random
from collections import OrderedDict, defaultdict

import numpy as np

import torch

from torch.optim import AdamW
# from util.multigpu_fused_adam import FusedAdam

from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import time
from os.path import join
from datetime import datetime

import util
from data.t5minimize_ner import minimize_language
from util.runner import Runner

from metrics import EREEvaluator


class NERRunner(Runner):

    def evaluate(self, model, tensor_examples, stored_info, step, predict=False):
        evaluator = EREEvaluator()

        eval_batch_size = 32
        if any(substr in self.config["plm_pretrained_name_or_path"].lower()\
           for substr in ["pp", "11b", "3b", "xl", "xxl", "large"]):
            eval_batch_size = 16

        util.runner.logger.info('Step %d: evaluating on %d samples with batch_size %d' % (
            step, len(tensor_examples), eval_batch_size))

        evalloader = DataLoader(
            tensor_examples, batch_size=eval_batch_size, shuffle=False, 
            num_workers=0,
            collate_fn=self.collate_fn, 
            pin_memory=True
        )
        model.eval()
        times = []
        total_examples = 0
        for i, (doc_keys, tensor_example) in enumerate(evalloader):
            example_gpu = {}

            for k, v in tensor_example.items():
                if v is not None:
                    example_gpu[k] = v.to(self.device)
            example_gpu['is_training'][:] = 0

            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=self.use_amp, dtype=torch.bfloat16
            ):
                start_t = time.time()
                output = model(**example_gpu)
                times.append(time.time() - start_t)

            for batch_id, doc_key in enumerate(doc_keys):
                gold_res = model.extract_gold_res_from_gold_annotation(
                    {k:v[batch_id] for k, v in tensor_example.items()}, 
                    stored_info['example'][doc_key]
                )
                start_t = time.time()
                decoded_results = model.decoding(
                    {k:v[batch_id] for k,v in output.items()}, 
                    stored_info['example'][doc_key]
                )
                times.append(time.time() - start_t)
                total_examples += 1

                decoded_results.update(
                    gold_res
                )
                evaluator.update(
                    **decoded_results
                )
                if predict:
                    util.runner.logger.info(stored_info['example'][doc_key])
                    util.runner.logger.info(decoded_results)

        total_time = sum(times)
        p,r,f = evaluator.get_prf()
        metrics = OrderedDict({
            'Eval_Ent_Precision': p[0] * 100,
            'Eval_Ent_Recall': r[0] * 100,
            'Eval_Ent_F1': f[0] * 100,
            'Eval_Time': total_time,
            'Eval_Examples': total_examples,
            'Eval_Examples_Per_Second': total_examples / total_time
        })
        for k,v in metrics.items():
            util.runner.logger.info('%s: %.4f'%(k, v))

        return f[0] * 100, metrics


# python run_ner.py t5_base 0
if __name__ == '__main__':
    config_file, config_name, gpu_id = sys.argv[1], sys.argv[2], 0
    config = util.initialize_config(config_name, config_file=config_file)

    data_dir = config["data_dir"]
    types_path = join(data_dir, config["types_path"])
    with open(types_path) as input_file:
        labels = json.load(input_file)
    entity_labels = {}

    for k in labels['entities'].keys():
        entity_labels[k] = len(entity_labels)

    stats = defaultdict(int)
    minimize_language(
        entity_labels,
        stats,
        data_dir,
        config["local_dir"],
        [
            config["train_path"],
            config["dev_path"],
            config["test_path"],
            config["test_path_seen"],
            config["test_path_unseen"],
        ]
    )
    print("stats:", stats)

    runner = NERRunner(
        config_file=config_file,
        config_name=config_name,
        gpu_id=gpu_id
    )
    model, _ = runner.initialize_model()
    runner.train(model, continued=False)
