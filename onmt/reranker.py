#!/usr/bin/env python
import configargparse

import os
import random
import torch

import onmt.opts as opts

from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from __future__ import print_function
import codecs
import math

from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.inputters as inputters
import onmt.decoders.ensemble


def build_reranker(opt, device_id, model, fields,
                  optim, data_type, model_saver=None):
    #train_loss = onmt.utils.loss.build_loss_compute(
    #    model, fields["tgt"].vocab, opt)
    #valid_loss = onmt.utils.loss.build_loss_compute(
    #    model, fields["tgt"].vocab, opt, train=False)

    trunc_size = opt.truncated_decoder  # Badly named...
    shard_size = opt.max_generator_batches
    norm_method = opt.normalization
    grad_accum_count = opt.accum_count
    n_gpu = opt.world_size
    if device_id >= 0:
        gpu_rank = opt.gpu_ranks[device_id]
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level
    n_hyps = opt.beam_size
    report_manager = onmt.utils.build_report_manager(opt)
    ranker = Reranker(model, optim, trunc_size,
                           shard_size, data_type, norm_method,
                           grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver, n_hyps=n_hyps)
    return ranker

class Ranker(object):
    def __init__(self, model, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None,n_hyps=5):
        # Basic attributes.
        self.model = model
        self.optim = optim
        self.trunc_size = trunc_size
        self.shard_size = shard_size
        self.data_type = data_type
        self.norm_method = norm_method
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.n_hyps = n_hyps

        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""
        self.criterion = nn.NLLLoss(
            ignore_index=self.padding_idx, reduce=False
        )

        # Set model in training mode.
        self.model.train()

    def get_scores(self, data_iter)
        self.model.eval()
        losses=[]
        for batch in data_iter:
            src = inputters.make_features(batch, 'src', self.data_type)
            _, src_lengths = batch.src
            tgt = inputters.make_features(batch, 'tgt')
            outputs, attns = self.model(src, tgt, src_lengths) #outputs is len X batch X rnn_size
            scores = self.generator(outputs)
            bsize = scores.size(1)
            gtruth = target.view(-1)
            loss = self.criterion(scores.view(-1,scores.size(2)), gtruth).view(-1, bsize).sum(0)
            losses.append(loss)
        return torch.cat(losses)

def main(opt, device_id):
    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    if opt.train_from:
        logger.info('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        fields = inputters.load_fields_from_vocab(
            checkpoint['vocab'], data_type=opt.data_type)

        # Load default opts values then overwrite it with opts from
        # the checkpoint. It's useful in order to re-train a model
        # after adding a new option (not set in checkpoint)
        dummy_parser = configargparse.ArgumentParser()
        opts.model_opts(dummy_parser)
        default_opt = dummy_parser.parse_known_args([])[0]

        model_opt = default_opt
        model_opt.__dict__.update(checkpoint['opt'].__dict__)
    else:
        checkpoint = None
        model_opt = opt
    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)
    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)
    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    src_features, tgt_features = _collect_report_features(fields)
    for j, feat in enumerate(src_features):
        logger.info(' * src feature %d size = %d'
                    % (j, len(fields[feat].vocab)))
    for j, feat in enumerate(tgt_features):
        logger.info(' * tgt feature %d size = %d'
                    % (j, len(fields[feat].vocab)))
    if use_gpu(opt):
        cur_device = "cuda"
    else:
        cur_device = "cpu"

    data = inputters. \
        build_dataset(fields,
                      opt.data_type,
                      src_path=opt.src,
                      tgt_path=opt.tgt)
    data_iter = inputters.OrderedIterator(
        dataset=data, device=cur_device,
        batch_size=opt.batch_size,
        Train=False,shuffle=False, sort=False, sort_within_batch=False)
    print len(data_iter.data())
    ranker = build_reranker(opt, device_id, model, fields, optim, opt.data_type, model_saver=model_saver)
    lmscores = ranker.get_scores(data_iter)
    mtscore_list=[]
    f = open(opt.scores,"r")
    for line in f:
      mtscores.append(float(line.strip().split()[0]))
    mtscores = torch.FloatTensor(mtscore_list)
    print mtscores, lmscores
    #trainer = build_trainer(opt, device_id, model, fields,
    #                        optim, data_type, model_saver=model_saver)

    # Do training.
    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
