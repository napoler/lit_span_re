# -*- coding: utf-8 -*-

"""

# 方案2 基于记忆的关系模型
当前模型可以基于三元组数据做预训练，主体设计借鉴sentent bert的孪生网络结构。
使用均值池化方案做。

https://github.com/lucidrains/x-transformers#augmenting-self-attention-with-persistent-memory




"""
from types import SimpleNamespace
from typing import Any, Optional

import pytorch_lightning as pl
import torch
# from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
# 自动停止
# https://pytorch-lightning.readthedocs.io/en/1.2.1/common/early_stopping.html
import torch.optim as optim
from performer_pytorch import Performer
# from .lr import CyclicCosineDecayLR,CyclicCosineDecayLRPlus
# from multiprocessing import Queue
# from .mask import autoMask
from tkitAutoMask import BertRandomMaskingScheme
# 引入贝叶斯优化
# from bayes_opt import BayesianOptimization
from tkitLr.CyclicCosineDecayLR import CyclicCosineDecayLR
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, f1, precision_recall
from transformers import AutoConfig, AutoModel, BertTokenizer
from einops import rearrange, reduce, repeat
from torch.nn.functional import one_hot
from torchmetrics.functional import accuracy, f1, precision_recall

from utils import mean_pooling
from inspect import isfunction


# from tkitAutoTokenizerPosition import AutoTokenizerPosition, autoBIO, autoSpan

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class MemoryReBlock(pl.LightningModule):
    """
    基础的命名实体
    简化版本丢弃矩阵相乘

    """

    def __init__(
            self,
            learning_rate=3e-4,
            T_max=5,
            hidden_size=256,
            vocab_size=21128,
            ignore_index=0,
            max_len=256,
            maxWordLen=65,
            num_labels=100,
            optimizer_name="AdamW", dropout=0.2,
            pretrained="uer/chinese_roberta_L-2_H-128",
            use_rnn=False,
            loss_alpha=0.5,
            labels=50,
            batch_size=2,
            trainfile="./data/train.pkt",
            valfile="./data/val.pkt",
            testfile="./data/test.pkt",
            num_memory_tokens=20,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)
        # pretrained="/data_200t/chenyaozu/data/base_model/chinese_roberta_L-4_H-512/"
        config = AutoConfig.from_pretrained(pretrained)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained)

        # config.num_labels=self.hparams.max_len
        config.output_attentions = True
        self.hparams.config = config
        # self.model = BertForPreTraining.from_pretrained(pretrained, config=config)
        self.model = AutoModel.from_pretrained(pretrained, config=config)

        # 定义孪生网络分类器
        self.fix = nn.Linear(config.hidden_size * 3, config.hidden_size)

        self.pre_classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(self.hparams.dropout),
            nn.LeakyReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(self.hparams.dropout),
            nn.LeakyReLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Dropout(self.hparams.dropout),
            nn.LeakyReLU(),
            nn.Linear(config.hidden_size ,config.hidden_size),
            nn.Dropout(self.hparams.dropout),
            nn.LeakyReLU(),
            nn.Linear(config.hidden_size, self.hparams.labels)
        )

        # memory tokens (like [cls]) from Memory Transformers paper
        # https://github.com/lucidrains/x-transformers/blob/b0c0ec9fad5eecdb8ed9b3c6e62547b6f06acd8d/x_transformers/x_transformers.py#L1006

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, config.hidden_size))

        # print("self.memory_tokens ", self.memory_tokens.size())

        self.loss_fc = nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def forward(self, input_ids, attention_mask=None, input_ids_b=None, attention_mask_b=None):

        x = self.model(input_ids=input_ids,
                       attention_mask=attention_mask)
        x_last_hidden_state = x.last_hidden_state
        emb_a = mean_pooling(x_last_hidden_state, attention_mask)

        b = self.model(input_ids=input_ids_b,
                       attention_mask=attention_mask_b)
        # x_last_hidden_state = b.last_hidden_state
        emb_b = mean_pooling(b.last_hidden_state, attention_mask)
        emb_diff = emb_a - emb_b
        emb = torch.cat((emb_a, emb_b, emb_diff.abs()), -1)
        emb = self.fix(emb)

        b, n = *emb_a.shape,
        # print("self.num_memory_tokens",self.memory_tokens.size())
        # print("b",b)
        num_mem = self.num_memory_tokens
        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            # print("mem",mem.size())
            emb = torch.cat((mem, emb.unsqueeze(1)), dim=1)
            # print("emb", emb.size())

        pooler = self.pre_classifier(emb)[:, :1]

        return pooler

    def getLoss(self, out, outType, out_lm=None, tag=None, tagtype=None, lm=None, attention_mask=None):
        """
        计算损失


        # attention_mask 用于限制文字长度
        """
        # B,L=out.size()

        # loss_fc=nn.CrossEntropyLoss()
        # # re_weights=torch.Tensor([0.01,0.5,0.5,0.5,0])
        # # loss_fc_re=nn.CrossEntropyLoss(ignore_index=0)
        # loss_fc_re=nn.CrossEntropyLoss()
        # loss_fc_mse=nn.MSELoss()

        # # 起始位置
        # num_labels = self.hparams.maxWordLen
        # w = num_labels*[1]
        # w[0] = 0.2
        # w = torch.Tensor(w).to(self.device)
        # Example of target with class indices
        # loss_fc = nn.CrossEntropyLoss(weight=w)
        loss_fc = nn.CrossEntropyLoss()

        # # 类型优化函数
        # num_labels = self.hparams.num_labels
        # w = num_labels*[1]
        # w[0] = 0.8
        # w = torch.Tensor(w).to(self.device)
        # # Example of target with class indices
        loss_fc_type = nn.CrossEntropyLoss()
        loss_fc_lm = nn.CrossEntropyLoss()

        if torch.sum(attention_mask) > 0:

            active_loss = attention_mask.view(-1) == 1
        else:
            active_loss = attention_mask.view(-1) >= -100

        loss_type = loss_fc_type(
            outType.view(-1, self.hparams.num_labels)[active_loss],
            tagtype.view(-1).long()[active_loss])

        loss_pos = loss_fc(
            out.view(-1, self.hparams.maxWordLen)[active_loss],
            tag.view(-1).long()[active_loss])
        if out_lm != None:
            loss_lm = loss_fc_lm(
                out_lm.view(-1, self.tokenizer.vocab_size),
                lm.view(-1).long())
            # 自动调整
            if loss_type > loss_pos:

                loss = loss_type * self.hparams.loss_alpha + loss_lm * (
                        1 - self.hparams.loss_alpha) * 1 / 100 + loss_pos * (
                               1 - self.hparams.loss_alpha) * 99 / 100
            else:
                loss = loss_pos * self.hparams.loss_alpha + loss_lm * (
                        1 - self.hparams.loss_alpha) * 1 / 100 + loss_type * (
                               1 - self.hparams.loss_alpha) * 99 / 100
            # loss = loss_type + loss_lm + loss_pos
        else:

            # loss = loss_pos * self.hparams.loss_alpha + loss_type * (
            #         1 - self.hparams.loss_alpha)
            # 验证时候采用1:1的比例
            loss = loss_pos + loss_type
            loss_lm = 0

        if loss >= 0:
            pass
        else:
            print(out, outType, tag, tagtype, attention_mask)
            exit()
        # loss=(loss1+loss2*1+loss_re*20)/22
        return loss, loss_pos, loss_type, loss_lm

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        # print(len(batch))
        input_ids, token_type_ids, attention_mask, tag, tagtype, re_spo = batch

        # # 修改序列长度
        # input_ids, token_type_ids, attention_mask, tag, tagtype = input_ids.view(-1,
        #                                                                          self.hparams.max_len), token_type_ids.view(
        #     -1, self.hparams.max_len), attention_mask.view(-1, self.hparams.max_len), tag.view(-1,
        #                                                                                        self.hparams.max_len), tagtype.view(
        #     -1, self.hparams.max_len)

        # # spanmask引入噪点
        args = SimpleNamespace(mask_ratio=0.15)
        #         input_ids,labels=self.tomask(input_ids)
        mk = BertRandomMaskingScheme(args, self.tokenizer.vocab_size, -100,
                                     self.tokenizer.mask_token_id)
        input_ids, labels, _ = mk.mask(input_ids.cpu())
        input_ids = torch.Tensor(input_ids).to(self.device).long()
        labels = torch.Tensor(labels).to(self.device).long()
        logits, outType, out_lm, loss_re = self(input_ids, token_type_ids, attention_mask, re_spo)

        # print("acc",acc)

        # 类型
        # 计算类型是 采用attention_mask作为筛选计算active_loss

        if torch.sum(attention_mask) > 0:

            active_loss = attention_mask.view(-1) == 1
        else:
            active_loss = attention_mask.view(-1) >= -100

        loss, loss_pos, loss_type, loss_lm = self.getLoss(
            logits, outType=outType, tag=tag, tagtype=tagtype, out_lm=out_lm, lm=labels,
            attention_mask=attention_mask)

        loss = loss_re + loss

        type_precision, type_recall = precision_recall(outType.argmax(dim=-1).view(-1)[active_loss],
                                                       tagtype.reshape(-1).long()[
                                                           active_loss], average='macro',
                                                       num_classes=self.hparams.num_labels)

        type_pred_f1 = f1(outType.argmax(dim=-1).view(-1)[active_loss], tagtype.reshape(-1).long()[active_loss],
                          num_classes=self.hparams.num_labels, average='macro')
        acc_type = accuracy(outType.argmax(-1).view(-1)[active_loss],
                            tagtype.int().view(-1)[active_loss])
        # 起始位置

        # # # 计算出所有类型结果
        # outType_index = outType.argmax(dim=-1)
        # outType_active = torch.where(
        #     outType_index > 0,
        #     outType_index,
        #     tagtype.long())
        # if torch.sum(outType_active.view(-1))>0:
        #     active_loss=outType_index.view(-1)>0
        # else:
        #     active_loss=active_loss

        lastOut = logits.argmax(dim=-1)

        precision, recall = precision_recall(lastOut.view(-1)[active_loss], tag.reshape(-1).long()[active_loss],
                                             average='macro', num_classes=self.hparams.maxWordLen)

        pred_f1 = f1(lastOut.view(-1)[active_loss], tag.reshape(-1).long()[active_loss],
                     num_classes=self.hparams.maxWordLen, average='macro')
        acc = accuracy(logits.argmax(-1).view(-1)[active_loss],
                       tag.int().view(-1)[active_loss])

        # lastType=outType.argmax(dim=-1)

        # logits.argmax(dim=-1)
        # 保留类型预测不为0的结果进行
        # last=torch.where(lastType==0,0,lastOut)

        # print(outType.argmax(-1))
        # print(tagtype)
        # print(acc,acc_type)
        # print("++"*20)

        metrics = {
            "train_precision_macro": precision,
            "train_recall_macro": recall,
            "train_f1_macro": pred_f1,
            "train_acc": acc,
            "train_type_precision_macro": type_precision,
            "train_type_recall_macro": type_recall,
            "train_type_f1_macro": type_pred_f1,
            "train_acc_type": acc_type,
            "train_loss_pos": loss_pos,
            "train_loss_type": loss_type,
            "train_loss_lm": loss_lm,
            "train_loss_full": (loss_pos + loss_type) / 2,
            "train_loss_re": loss_re
        }
        # print("metrics",metrics)
        self.log_dict(metrics)
        self.log('train_loss', loss)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     # training_step defined the train loop.
    #     # It is independent of forward
    #     # print(len(batch))
    #     input_ids, token_type_ids, attention_mask, tag, tagtype = batch
    #
    #     input_ids, token_type_ids, attention_mask, tag, tagtype = input_ids.view(-1,
    #                                                                              self.hparams.max_len), token_type_ids.view(
    #         -1, self.hparams.max_len), attention_mask.view(-1, self.hparams.max_len), tag.view(-1,
    #                                                                                            self.hparams.max_len), tagtype.view(
    #         -1, self.hparams.max_len)
    #
    #     logits, outType, out_lm = self(input_ids, token_type_ids, attention_mask)
    #
    #     # print("acc",acc)
    #
    #     # 类型
    #     # 计算类型是 采用attention_mask作为筛选计算active_loss
    #
    #     if torch.sum(attention_mask) > 0:
    #
    #         active_loss = attention_mask.view(-1) == 1
    #     else:
    #         active_loss = attention_mask.view(-1) >= -100
    #
    #     loss, loss_pos, loss_type, loss_lm = self.getLoss(
    #         logits, outType=outType, tag=tag, tagtype=tagtype, attention_mask=attention_mask, out_lm=None, lm=None)
    #
    #     type_precision, type_recall = precision_recall(outType.argmax(dim=-1).view(-1)[active_loss],
    #                                                    tagtype.reshape(-1).long()[
    #                                                        active_loss], average='macro',
    #                                                    num_classes=self.hparams.num_labels)
    #
    #     type_pred_f1 = f1(outType.argmax(dim=-1).view(-1)[active_loss], tagtype.reshape(-1).long()[active_loss],
    #                       num_classes=self.hparams.num_labels, average='macro')
    #
    #     acc_type = accuracy(outType.argmax(-1).view(-1)[active_loss],
    #                         tagtype.int().view(-1)[active_loss])
    #     # 起始位置
    #
    #     # # # 计算出所有类型结果
    #     # outType_index = outType.argmax(dim=-1)
    #     # outType_active = torch.where(
    #     #     outType_index > 0,
    #     #     outType_index,
    #     #     tagtype.long())
    #     # if torch.sum(outType_active.view(-1))>0:
    #     #     active_loss=outType_index.view(-1)>0
    #     # else:
    #     #     active_loss=active_loss
    #
    #     lastOut = logits.argmax(dim=-1)
    #
    #     precision, recall = precision_recall(lastOut.view(-1)[active_loss], tag.reshape(-1).long()[active_loss],
    #                                          average='macro', num_classes=self.hparams.maxWordLen)
    #
    #     pred_f1 = f1(lastOut.view(-1)[active_loss], tag.reshape(-1).long()[active_loss],
    #                  num_classes=self.hparams.maxWordLen, average='macro')
    #     pos_acc = accuracy(logits.argmax(-1).view(-1)[active_loss],
    #                        tag.int().view(-1)[active_loss])
    #
    #     # lastType=outType.argmax(dim=-1)
    #
    #     # logits.argmax(dim=-1)
    #     # 保留类型预测不为0的结果进行
    #     # last=torch.where(lastType==0,0,lastOut)
    #
    #     metrics = {
    #         "val_f1_full": (pred_f1 + type_pred_f1) / 2,
    #         "val_precision_macro": precision,
    #         "val_recall_macro": recall,
    #         # "val_"
    #         "val_f1_macro": pred_f1,
    #         "val_acc": pos_acc,
    #         "val_acc_type": acc_type,
    #         "val_type_precision_macro": type_precision,
    #         "val_type_recall_macro": type_recall,
    #         "val_type_f1_macro": type_pred_f1,
    #         "val_loss_pos": loss_pos,
    #         "val_loss_type": loss_type,
    #         "val_loss": loss,
    #     }
    #     # print("metrics",metrics)
    #     self.log_dict(metrics)
    #     # self.log('train_loss', loss)
    #     return metrics
    #
    # def test_step(self, batch, batch_idx):
    #     # training_step defined the train loop.
    #     # It is independent of forward
    #     input_ids, token_type_ids, attention_mask, tag, tagtype = batch
    #
    #     input_ids, token_type_ids, attention_mask, tag, tagtype = input_ids.view(-1,
    #                                                                              self.hparams.max_len), token_type_ids.view(
    #         -1, self.hparams.max_len), attention_mask.view(-1, self.hparams.max_len), tag.view(-1,
    #                                                                                            self.hparams.max_len), tagtype.view(
    #         -1, self.hparams.max_len)
    #
    #     logits, outType, out_lm = self(input_ids, token_type_ids, attention_mask)
    #
    #     loss, loss_pos, loss_type, loss_lm = self.getLoss(
    #         logits, outType=outType, tag=tag, tagtype=tagtype, attention_mask=attention_mask, out_lm=None, lm=None)
    #
    #     active_loss = attention_mask.view(-1) > -100
    #     testf = open("data/test_pos_base_new_test.txt", "a+")
    #
    #     with open("data/test_pos_base_new.txt", "a+") as f:
    #         for i, (x, out_pos, y_pos, out_type, y_type, masks) in enumerate(zip(input_ids.tolist(),
    #                                                                              logits.argmax(
    #                                                                                  dim=-1).tolist(),
    #                                                                              tag.tolist(),
    #                                                                              outType.argmax(
    #                                                                                  dim=-1).tolist(),
    #                                                                              tagtype.tolist(),
    #                                                                              attention_mask.tolist(),
    #                                                                              )):
    #             #             print(p,y)
    #             words = self.tokenizer.convert_ids_to_tokens(x)
    #             #                 print(words)
    #             word_dict = {}
    #             word_y_dict = {}
    #             f.write("\n\n\n")
    #             f.write("######" * 20)
    #             f.write("".join(words))
    #             f.write("\n预测位置")
    #             f.write(str(out_pos) + "\n实际标注位置" + str(y_pos))
    #             f.write("\n")
    #             for ii, (pit, yit, tx, ty, mask) in enumerate(zip(out_pos, y_pos, out_type, y_type, masks)):
    #                 #                     if pit!=0 and  yit!=0 and mask!=0:
    #                 if mask != 0:
    #                     testf.write(words[ii] + "," + str(int(yit)) + "," + str(int(ty)) +
    #                                 "\n")
    #                 if yit != 0:
    #                     word_y_dict[ii] = words[ii:ii + int(yit)]
    #                 if mask != 0 and (pit != 0 or yit != 0):
    #                     f.write("\n" * 3)
    #                     f.write("******New*******")
    #                     if int(pit) == int(yit):
    #                         f.write("\n")
    #                         f.write("预测成功###########")
    #                     f.write("\n")
    #
    #                     f.write("类型yu-or):" + str(tx) + "=>" + str(ty))
    #                     f.write("\n")
    #                     f.write("预测:")
    #                     f.write("---")
    #
    #                     word_dict[ii] = words[ii:ii + int(pit)]
    #
    #                     f.write(" ".join(words[ii:ii + int(pit)]))
    #                     f.write("---")
    #                     f.write("\n")
    #
    #                     f.write("标记:")
    #                     #                         f.write(self.hparams.labels[int(ty)])
    #                     # f.write()
    #                     f.write("--")
    #                     f.write(" ".join(words[ii:ii + int(yit)]))
    #             # f.write("####"*10)
    #             # f.write("\n"*5)
    #
    #     metrics = {"test_loss": loss}
    #     self.log_dict(metrics)
    #     return metrics

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # training_step defined the train loop.
        # It is independent of forward
        input_ids, token_type_ids, attention_mask = batch

        input_ids, token_type_ids, attention_mask = input_ids.view(-1, self.hparams.max_len), token_type_ids.view(
            -1, self.hparams.max_len), attention_mask.view(-1, self.hparams.max_len)

        logits, outType, _ = self(input_ids, token_type_ids, attention_mask)
        # active_loss = attention_mask.view(-1) > -100
        # testf = open("data/test_pos_base_new_testpredict_step.txt", "a+")
        outdata = []
        with open("data/test_pos_base_predict_step.txt", "a+") as f:
            for i, (x, out_pos, out_type, masks) in enumerate(zip(input_ids.tolist(),
                                                                  logits.argmax(
                                                                      dim=-1).tolist(),
                                                                  outType.argmax(
                                                                      dim=-1).tolist(),
                                                                  attention_mask.tolist(),
                                                                  )):
                #             print(p,y)
                words = self.tokenizer.convert_ids_to_tokens(x)
                #                 print(words)
                word_dict = {}
                word_y_dict = {}

                for ii, (pit, tx, mask) in enumerate(zip(out_pos, out_type, masks)):
                    #                     if pit!=0 and  yit!=0 and mask!=0:
                    # if mask != 0:
                    #     testf.write(words[ii] + "," + str(int(yit)) + "," + str(int(ty)) +
                    #                 "\n")
                    if pit != 0 and mask != 0 and tx != 0:
                        word_y_dict[ii] = words[ii:ii + int(pit)]
                        word_y_dict[ii] = tx

                outdata.append({"words": words, "word_dict": word_dict, "word_y_dict": word_y_dict})
        return outdata

    def train_dataloader(self):
        train = torch.load(self.hparams.trainfile)
        return DataLoader(train, batch_size=int(self.hparams.batch_size), num_workers=24, pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        val = torch.load(self.hparams.valfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=24, pin_memory=True)

    def test_dataloader(self):
        val = torch.load(self.hparams.testfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=24, pin_memory=True)

    #     def configure_optimizers(self):
    #         """优化器 自动优化器"""
    #         optimizer = getattr(optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.learning_rate*2)
    #         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=10, min_lr=5e-6)
    #         # scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
    #         # scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])

    # #
    #         lr_scheduler={
    #             'scheduler': scheduler,
    #             'interval': 'step',
    #             'frequency': 100,
    #             'name':"lr_scheduler",
    #             'monitor': 'train_loss', #监听数据变化
    #             'strict': True,
    #         }
    # #         return [optimizer], [lr_scheduler]
    #         return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def configure_optimizers(self):
        """优化器 自动优化器"""
        optimizer = getattr(optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)

        # optimizer = getattr(torch.optim,self.hparams.optimizer_name)([
        #         # {'params': self.parameters(), 'lr': self.hparams.learning_rate},

        #         {'params': filter(lambda p: p.requires_grad, self.parameters()), 'lr': self.hparams.learning_rate},

        #         # {'params': filter(lambda p: p.requires_grad, self.Diagonal_optimization.parameters()), 'lr': self.hparams.learning_rate},

        #         # {'params': filter(lambda p: p.requires_grad, self.model.parameters()), 'lr': self.hparams.learning_rate}
        #        ]
        #         )
        #         使用自适应调整模型
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=500,factor=0.8,verbose=True)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2, eta_min=0,
        #                                                                  verbose=False)
        scheduler = CyclicCosineDecayLR(optimizer,
                                        init_decay_epochs=1000,
                                        min_decay_lr=1e-8,
                                        restart_interval=500,
                                        restart_lr=self.hparams.learning_rate / 2,
                                        restart_interval_multiplier=1.1,
                                        warmup_epochs=1000,
                                        warmup_start_lr=self.hparams.learning_rate / 20)
        #
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'name': "lr_scheduler",
            'monitor': 'train_loss_full',  # 监听数据变化
            'strict': True,
        }
        #         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


if __name__ == '__main__':
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained('uer/chinese_roberta_L-2_H-128')
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt", max_length=128, pad_token="max_length")
    inputsB = tokenizer("Hello, my dog is cute", return_tensors="pt", max_length=128, pad_token="max_length")
    # src = torch.randint(0, 256, (4, 128))
    print(inputs)
    # print(src.shape)
    model = MemoryReBlock()
    # print(model)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    input_ids_b = inputsB['input_ids']
    attention_mask_b = inputsB['attention_mask']
    out = model(input_ids, attention_mask, input_ids_b, attention_mask_b)
    print(out.size())
    # print(model)

    pass
