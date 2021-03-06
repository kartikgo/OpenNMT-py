#!/usr/bin/env python
from __future__ import print_function
import configargparse

import os
import random
import torch

import onmt.opts as opts
from onmt.utils.misc import use_gpu
from onmt.inputters.inputter import build_dataset_iter, lazily_load_dataset, \
    _load_fields, _collect_report_features
from onmt.model_builder import build_model
from onmt.utils.optimizers import build_optim
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
import codecs
import math

import torch.nn as nn
import torch.nn.functional as F
from itertools import count
from onmt.utils.misc import tile

import onmt.model_builder
import onmt.inputters as inputters
import onmt.decoders.ensemble


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec

def _check_save_model_path(opt):
    save_model_path = os.path.abspath(opt.save_model)
    model_dirname = os.path.dirname(save_model_path)
    if not os.path.exists(model_dirname):
        os.makedirs(model_dirname)

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
    ranker = Ranker(opt, model, optim, trunc_size,
                           shard_size, data_type, norm_method,
                           grad_accum_count, n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver, n_hyps=n_hyps, 
                           padding_idx = fields["tgt"].vocab.stoi[inputters.PAD_WORD])
    return ranker

class Ranker(object):
    def __init__(self, opt, model, optim,
                 trunc_size=0, shard_size=32, data_type='text',
                 norm_method="sents", grad_accum_count=1, n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None, n_hyps=10, padding_idx=None):
        # Basic attributes.
        self.model = model
        self.optim = optim
        self.opt = opt
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
        self.padding_idx = padding_idx
        self.marginloss = nn.MarginRankingLoss(margin=1,reduction='none')
        assert grad_accum_count > 0
        if grad_accum_count > 1:
            assert(self.trunc_size == 0), \
                """To enable accumulated gradients,
                   you must disable target sequence truncating."""
        self.criterion = nn.NLLLoss(
            ignore_index=self.padding_idx, reduction='none'
        )

        # Set model in training mode.
        self.model.train()
        self.model.generator.train()

    def get_best_hyps(self, data_iter):
        self.model.eval()
        self.model.generator.eval()
        bests = []
        for i,batch in enumerate(data_iter):
            tgt = inputters.make_features(batch,'tgt')
            tgt = tgt.transpose(0,1).contiguous().squeeze() #hopefully batch,len
            gt = tgt[0]
            hyps = tgt[1:]
            best = ((torch.eq(gt,hyps).sum(1).topk(1))[1]+1).squeeze().item()
            bests.append(best)
        inds = [a*(self.n_hyps+1) for a in list(range(len(bests)))]
        print(sum([0 if (a==0) else 1 for a in bests]))
        #print(inds)
        final = [a+b for (a,b) in zip(bests,inds)]
        return final

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _compute_loss(self, refs, modelscores, bleuscores,mask):
        if (bleuscores is None):
            bleuscores = 0.0
        bsize = refs.size(0)
        y = torch.ones(bsize)
        if(self.gpu_rank==0):
            y = y.cuda()
        #if (liang_margin):
        weighted_ref_loss =(1.0 - bleuscores)*(mask*self.marginloss(refs, modelscores, y))
        bleuscores = mask*bleuscores
        best_bl,inds = torch.topk(bleuscores.view(-1,self.n_hyps+1),1)
        diff_bleu = (best_bl - bleuscores.view(-1, self.n_hyps+1)).view(-1)
        cand_loss = mask*diff_bleu*torch.nn.functional.relu(diff_bleu*10 +  modelscores.view(-1, self.n_hyps + 1) - modelscores.view(-1, self.n_hyps+1).gather(1, inds)).view(-1).contiguous()
        #print(refs.tolist(), modelscores.tolist(), bleuscores.tolist(), weighted_ref_loss.tolist(), cand_loss.tolist())
        return (cand_loss + weighted_ref_loss)
        #elif ()

    def train(self, train_iter, valid_iter, train_steps, valid_steps, train_ext_score = None, valid_ext_score = None, glob= False, mul=0.0, train_bleu = None):
        logger.info('Start training...')

        step = self.optim._step + 1
        accum = 0
        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)
        normalization = 0
        step = 0
        if not(train_ext_score is None):
            extscores = torch.tensor(train_ext_score)
        else:
            extscores = None
        if not(train_bleu is None):
            bleuscores = torch.tensor(train_bleu)
        else:
            bleuscores = None
        while (step  < train_steps):
            st=0
            for i,batch in enumerate(train_iter):
                self.model.zero_grad()
                src = inputters.make_features(batch, 'src', self.data_type)
                _, src_lengths = batch.src
                tgt = inputters.make_features(batch, 'tgt')
                outputs, attns = self.model(src, tgt, src_lengths) #outputs is len X batch X rnn_size
                bsize = outputs.size(1)
                #print(bsize)
                if not(extscores is None):
                    #extscore = extscores[i*bsize: (i+1)*bsize]
                    extscore = extscores[st: st + bsize]
                else:
                    extscore = torch.zeros(bsize)
                if not(bleuscores is None):
                    #bleuscore = bleuscores[i*bsize: (i+1)*bsize]
                    bleuscore = bleuscores[st: st+bsize]
                else:
                    bleuscore = torch.zeros(bsize)
                st = st+bsize
                mask = torch.ones(bsize)
                ref_inds = (self.n_hyps+1)*torch.tensor(list(range(int(bsize/(self.n_hyps + 1)))))
                mask[ref_inds] = 0.0
                if (self.gpu_rank==0):
                    ref_inds= ref_inds.cuda()
                    mask = mask.cuda()
                    extscore = extscore.cuda()
                    bleuscore = bleuscore.cuda()
                assert (bsize % (self.n_hyps+1)==0)
                gtruth = tgt[1:].view(-1)
                wts = (gtruth != self.padding_idx).float()
                lgts = wts.view(-1, bsize).sum(0)
                if (glob):
                    scores = self.model.generator[0](outputs)
                else:
                    scores = self.model.generator(outputs)
                all_scores  = ((scores.view(-1, scores.size(2)).gather(1, gtruth.unsqueeze(1))).squeeze()*wts).view(-1, bsize).sum(0).squeeze()
                #print(all_scores.tolist(), extscore.tolist())
                all_scores = (all_scores + mul*extscore).div(lgts)
                refs = torch.index_select(all_scores, 0, ref_inds).unsqueeze(1).expand(ref_inds.size(0), self.n_hyps + 1).contiguous().view(-1)        
                assert refs.size()==all_scores.size()
                #loss = (1.0 - bleuscore)*(mask*self.marginloss(refs, all_scores, y))
                loss = self._compute_loss(refs, all_scores, bleuscore, mask)
                #loss = loss.div(bleuscore+1e-10)
                normalization = batch.batch_size#/(self.n_hyps + 1)
                (loss.sum()).div(float(normalization)).backward()
                self.optim.step()
                batch_stats = onmt.utils.Statistics(loss.sum().item(), float(normalization), 0, 0.0, 0.0)
                report_stats.update(batch_stats)
                total_stats.update(batch_stats)
                step += 1
                report_stats = self._maybe_report_training(
                    step, train_steps,
                    self.optim.learning_rate,
                    report_stats)
                normalization =0 
                if (step % valid_steps == 0):
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: validate step %d'
                                    % (self.gpu_rank, step))
                    #valid_iter = valid_iter_fct()
                    valid_stats = self.validate(valid_iter, valid_ext_score, glob, mul)
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: gather valid stat \
                                    step %d' % (self.gpu_rank, step))
                    valid_stats = self._maybe_gather_stats(valid_stats)
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: report stat step %d'
                                    % (self.gpu_rank, step))
                    self._report_step(self.optim.learning_rate,
                                      step, valid_stats=valid_stats)
                    valid_stats = self.validate(train_iter, train_ext_score, glob, mul, train=True)
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: gather valid stat \
                                    step %d' % (self.gpu_rank, step))
                    valid_stats = self._maybe_gather_stats(valid_stats)
                    if self.gpu_verbose_level > 0:
                        logger.info('GpuRank %d: report stat step %d'
                                    % (self.gpu_rank, step))
                    self._report_step(self.optim.learning_rate,
                                      step, valid_stats=valid_stats)
                if self.gpu_rank == 0:
                    self._maybe_save(step)
            if self.gpu_verbose_level > 0:
                logger.info('GpuRank %d: we completed an epoch \
                            at step %d' % (self.gpu_rank, step))
                        
    def validate(self, valid_iter, valid_ext_score, glob= False, mul =0.0, train = False):
        self.model.eval() 
        self.model.generator.eval() 
        scores_list=[]
        stats = onmt.utils.Statistics()
        if not(valid_ext_score is None):
            extscores = torch.tensor(valid_ext_score)
        else:
            extscores = None
        st = 0
        for i,batch in enumerate(valid_iter):
            if (train and (i> 100)):
                break
            src = inputters.make_features(batch, 'src', self.data_type)
            _, src_lengths = batch.src
            tgt = inputters.make_features(batch, 'tgt')
            outputs, attns = self.model(src, tgt, src_lengths) #outputs is len X batch X rnn_size
            bsize = outputs.size(1)
            if not(extscores is None):
                #extscore = extscores[i*bsize: (i+1)*bsize]
                extscore = extscores[st: st+bsize]
            else:
                extscore = torch.zeros(bsize)
            st = st+bsize
            assert (bsize % (self.n_hyps+1)==0)
            mask = torch.ones(bsize)
            #ref_inds = torch.tensor(list(range(int(bsize/(self.n_hyps + 1)))))
            ref_inds = (self.n_hyps+1)*torch.tensor(list(range(int(bsize/(self.n_hyps + 1)))))
            mask[ref_inds] = 0.0
            if (self.gpu_rank==0):
                ref_inds= ref_inds.cuda()
                mask = mask.cuda()
                extscore = extscore.cuda()
            gtruth = tgt[1:].view(-1)
            wts = (gtruth != self.padding_idx).float()
            lgts = wts.view(-1, bsize).sum(0)
            if (glob):
                scores = self.model.generator[0](outputs)
            else:
                scores = self.model.generator(outputs)
            all_scores  = ((scores.view(-1, scores.size(2)).gather(1, gtruth.unsqueeze(1))).squeeze()*wts).view(-1, bsize).sum(0).squeeze()
            all_scores= all_scores.div(lgts) + mul*extscore
            refs = torch.index_select(all_scores, 0, ref_inds).unsqueeze(1).expand(ref_inds.size(0), self.n_hyps + 1).contiguous().view(-1)        
            assert refs.size()==all_scores.size()
            #loss = mask*self.marginloss(refs, all_scores, y)
            bleuscore = None
            loss = self._compute_loss(refs, all_scores, bleuscore, mask)
            normalization = batch.batch_size/(self.n_hyps + 1)
            batch_stats = onmt.utils.Statistics(loss.sum().item(), float(normalization), 0, 0.0, 0.0)
            stats.update(batch_stats)
            scores_list += all_scores.tolist()
        scores = torch.tensor(scores_list)
        if (self.gpu_rank==0):
            scores = scores.cuda()
        selected = self.rank(scores)
        f = open(self.opt.tgt+".valid",'r').readlines()
        cands=[]
        for ind in selected:
            cands.append(f[ind])
        outf = open(self.opt.output,'w+')
        for line in cands:
            outf.write(line)
            outf.flush()
        msg = report_bleu(outf, "./ted_data/valid.de-en.en")
        logger.info(msg)
        outf.close()
        self.model.train()
        self.model.generator.train()
        valid_stats = self._maybe_gather_stats(stats)
        self._report_step(self.optim.learning_rate,
                          1, valid_stats=valid_stats)
        return stats

    def get_scores(self, data_iter, glob = False):
        self.model.eval()
        #self.model.generator.eval()
        losses=[]
        for i,batch in enumerate(data_iter):
            #print(i)
            src = inputters.make_features(batch, 'src', self.data_type)
            _, src_lengths = batch.src
            tgt = inputters.make_features(batch, 'tgt')
            outputs, attns = self.model(src, tgt, src_lengths) #outputs is len X batch X rnn_size
            if (glob):
                scores = self.model.generator[0](outputs)
                bsize = scores.size(1)
                gtruth = tgt[1:].view(-1)
                wts = (gtruth != self.padding_idx).float()
                loss = ((scores.view(-1, scores.size(2)).gather(1, gtruth.unsqueeze(1))).squeeze()*wts).view(-1, bsize).sum(0)
                #loss = (scores.view(-1,scores.size(2)), gtruth).view(-1, bsize).sum(0)
                losses += loss.squeeze().tolist()
            else:
                scores = self.model.generator(outputs)
                bsize = scores.size(1)
                gtruth = tgt[1:].view(-1)
                wts = (gtruth != self.padding_idx).float()
                #loss = -1.0*self.criterion(scores.view(-1,scores.size(2)), gtruth).view(-1, bsize).sum(0)
                loss = ((scores.view(-1, scores.size(2)).gather(1, gtruth.unsqueeze(1))).squeeze()*wts).view(-1, bsize).sum(0)
                losses += loss.squeeze().tolist()
        return losses
        #return None

    def rank(self, scores):
        tot_scores = scores
        hyp_scores = torch.stack([a[1:] for a in list(tot_scores.split(self.n_hyps+1))])
        raw_rank = (torch.topk(hyp_scores,1)[1] + 1).squeeze().tolist()
        inds = [a*(self.n_hyps+1) for a in list(range(len(raw_rank)))]
        final = [a+b for (a,b) in zip(raw_rank,inds)]
        return final

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return onmt.utils.Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)

def training_opt_postprocessing(opt, device_id):
    if opt.word_vec_size != -1:
        opt.src_word_vec_size = opt.word_vec_size
        opt.tgt_word_vec_size = opt.word_vec_size

    if opt.layers != -1:
        opt.enc_layers = opt.layers
        opt.dec_layers = opt.layers

    if opt.rnn_size != -1:
        opt.enc_rnn_size = opt.rnn_size
        opt.dec_rnn_size = opt.rnn_size
        if opt.model_type == 'text' and opt.enc_rnn_size != opt.dec_rnn_size:
            raise AssertionError("""We do not support different encoder and
                                 decoder rnn sizes for translation now.""")

    opt.brnn = (opt.encoder_type == "brnn")

    if opt.rnn_type == "SRU" and not opt.gpu_ranks:
        raise AssertionError("Using SRU requires -gpu_ranks set.")

    if torch.cuda.is_available() and not opt.gpu_ranks:
        logger.info("WARNING: You have a CUDA device, \
                    should run with -gpu_ranks")

    if opt.seed > 0:
        torch.manual_seed(opt.seed)
        # this one is needed for torchtext random call (shuffled iterator)
        # in multi gpu it ensures datasets are read in the same order
        random.seed(opt.seed)
        # some cudnn methods can be random even after fixing the seed
        # unless you tell it to be deterministic
        torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        if opt.seed > 0:
            # These ensure same initialization in multi gpu mode
            torch.cuda.manual_seed(opt.seed)

    return opt

def report_bleu(out_file, tgt_path):
    import subprocess
    base_dir = os.path.abspath(__file__ + "/../../..")
    # Rollback pointer to the beginning.
    out_file.seek(0)
    print()

    res = subprocess.check_output("perl ./multi-bleu.perl %s"
                                  % (tgt_path),
                                  stdin=out_file,
                                  shell=True).decode("utf-8")

    msg = ">> " + res.strip()
    return msg

def main(opt, device_id):
    opt = training_opt_postprocessing(opt, device_id)
    init_logger(opt.log_file)
    # Load checkpoint if we resume from a previous training.
    #if opt.train_from:
    # Load default opts values then overwrite it with opts from
    # the checkpoint. It's useful in order to re-train a model
    # after adding a new option (not set in checkpoint)
    dummy_parser = configargparse.ArgumentParser()
    opts.model_opts(dummy_parser)
    default_opt = dummy_parser.parse_known_args([])[0]
    logger.info('Loading checkpoint from %s' % opt.train_from)
    if not(opt.mtrerank):
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        fields = inputters.load_fields_from_vocab(
            checkpoint['vocab'], data_type=opt.data_type)
        model_opt = default_opt
        model_opt.__dict__.update(checkpoint['opt'].__dict__)
        model = build_model(model_opt, opt, fields, checkpoint)
    else:
        checkpoint = torch.load(opt.models[0], map_location=lambda storage, loc: storage)
        fields = inputters.load_fields_from_vocab(
            checkpoint['vocab'], data_type=opt.data_type)
        model_opt = default_opt
        model_opt.__dict__.update(checkpoint['opt'].__dict__)
        model = build_model(model_opt, opt, fields, checkpoint)
    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)
    _check_save_model_path(opt)
    # Build optimizer.
    optim = build_optim(model, opt, checkpoint)
    # Build model saver
    #model_saver = build_model_saver(model_opt, opt, model, fields, optim)
    model_saver= None
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
    #print(opt.src, opt.tgt)
    if not(opt.mtrerank):
        data = inputters. \
            build_dataset(fields,
                          opt.data_type,
                          src_path=opt.src+'.valid',
                          src_data_iter=None,
                          tgt_path=opt.tgt+'.valid',
                          tgt_data_iter=None,
                          src_dir=None,
                          sample_rate=opt.sample_rate,
                          window_size=opt.window_size,
                          window_stride=opt.window_stride,
                          window=opt.window,
                          use_filter_pred=False,
                          image_channel_size=opt.image_channel_size)
        valid_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=opt.batch_size,
            train=False,shuffle=False, sort=False, sort_within_batch=False)
        data = inputters. \
            build_dataset(fields,
                          opt.data_type,
                          src_path=opt.src+'.train',
                          src_data_iter=None,
                          tgt_path=opt.tgt+'.train',
                          tgt_data_iter=None,
                          src_dir=None,
                          sample_rate=opt.sample_rate,
                          window_size=opt.window_size,
                          window_stride=opt.window_stride,
                          window=opt.window,
                          use_filter_pred=False,
                          image_channel_size=opt.image_channel_size)
        train_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=opt.batch_size,
            train=False,shuffle=False, sort=False, sort_within_batch=False)
    else:
        data = inputters. \
            build_dataset(fields,
                          opt.data_type,
                          src_path=opt.mt_src+'.valid',
                          src_data_iter=None,
                          tgt_path=opt.tgt+'.valid',
                          tgt_data_iter=None,
                          src_dir=None,
                          sample_rate=opt.sample_rate,
                          window_size=opt.window_size,
                          window_stride=opt.window_stride,
                          window=opt.window,
                          use_filter_pred=False,
                          image_channel_size=opt.image_channel_size)
        valid_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=opt.batch_size,
            train=False,shuffle=False, sort=False, sort_within_batch=False)
        data = inputters. \
            build_dataset(fields,
                          opt.data_type,
                          src_path=opt.mt_src+'.train',
                          src_data_iter=None,
                          tgt_path=opt.tgt+'.train',
                          tgt_data_iter=None,
                          src_dir=None,
                          sample_rate=opt.sample_rate,
                          window_size=opt.window_size,
                          window_stride=opt.window_stride,
                          window=opt.window,
                          use_filter_pred=False,
                          image_channel_size=opt.image_channel_size)
        train_iter = inputters.OrderedIterator(
            dataset=data, device=cur_device,
            batch_size=opt.batch_size,
            train=False,shuffle=False, sort=False, sort_within_batch=False)
    btchs = 0
    ranker = build_reranker(opt, device_id, model, fields, optim, opt.data_type, model_saver=model_saver)
    train_extscore_list=[]
    valid_extscore_list=[]
    bleu_list = []
    #selected = ranker.get_best_hyps(data_iter)
    #print(inds)
    f = open("10best_nounk.bleu.train","r")
    for line in f:
      bleu_list.append(float(line.strip().split()[0]))
    f.close()
    f = open(opt.scores+".train","r")
    for line in f:
      train_extscore_list.append(float(line.strip().split()[0]))
    f.close()
    f = open(opt.scores+".valid","r")
    for line in f:
      valid_extscore_list.append(float(line.strip().split()[0]))
    f.close()
    if (opt.nobleu):
        bleu_list = None
    if (opt.noext):
        train_extscore_list = None
        valid_extscore_list = None
    #stats = ranker.validate(data_iter, mtscore_list, glob= opt.glob, mul = opt.mt_mul)
    ranker.train(train_iter, valid_iter, opt.train_steps, opt.valid_steps, train_extscore_list, valid_extscore_list, glob= opt.glob, mul = opt.mt_mul, train_bleu = bleu_list)
    '''mtscores = torch.cuda.FloatTensor(mtscore_list)
    lmscore_list = ranker.get_scores(data_iter, glob= opt.glob)
    lmscores = torch.cuda.FloatTensor(lmscore_list)
    print(mtscores.mean().item())
    print(lmscores.mean().item())
    selected = ranker.combine_rank(mtscores, lmscores, opt.mt_mul)
    f = open(opt.tgt,'r').readlines()
    cands=[]
    for ind in selected:
        cands.append(f[ind])
    outf = open(opt.output,'w+')
    for line in cands:
        outf.write(line)
        outf.flush()
    msg = report_bleu(outf, "./ted_data/valid.de-en.en")
    logger.info(msg)
    outf.close()'''

    # Do training.
    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')

if __name__ == "__main__":
    parser = configargparse.ArgumentParser(
        description='train.py',
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

    opts.add_md_help_argument(parser)
    opts.model_opts(parser)
    opts.train_opts(parser)

    opt = parser.parse_args()
    main(opt)
