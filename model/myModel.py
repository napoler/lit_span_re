# -*- coding: utf-8 -*-


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


class myModel(pl.LightningModule):
    """
    基础的命名实体
    简化版本丢弃矩阵相乘

    """

    def __init__(
            self, learning_rate=3e-4, T_max=5,
            hidden_size=256,
            vocab_size=21128,
            ignore_index=0, max_len=256, maxWordLen=65, num_labels=100,
            optimizer_name="AdamW", dropout=0.2,
            pretrained="uer/chinese_roberta_L-2_H-128",
            use_rnn=False,
            loss_alpha=0.5,
            labels=50,
            batch_size=2, trainfile="./data/train.pkt", valfile="./data/val.pkt", testfile="./data/test.pkt", **kwargs):
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
        self.lmhead = nn.Sequential(
            # nn.Dropout(self.hparams.dropout),
            nn.Linear(config.hidden_size, self.tokenizer.vocab_size),
        )
        # Rnn use_rnn
        if self.hparams.use_rnn == True:
            self.rnn = nn.GRU(config.hidden_size, config.hidden_size, num_layers=2, batch_first=True, dropout=dropout,
                              bidirectional=True)
            self.fix = nn.Linear(config.hidden_size * 2, config.hidden_size)

        # 用于输出词性分类
        self.out_type = nn.Sequential(
            nn.Linear(config.hidden_size, self.hparams.hidden_size),
            # nn.Dropout(self.hparams.dropout),
            # Performer(
            #     dim=256,
            #     depth=3,
            #     heads=8,
            #     dim_head=64,
            #     local_window_size=128,
            #     kernel_fn=nn.Tanh(),
            #     ff_dropout=self.hparams.dropout,
            #     attn_dropout=self.hparams.dropout,
            #     shift_tokens=True,
            # ),
            nn.LeakyReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_size, self.hparams.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.hidden_size, self.hparams.num_labels)
        )

        # 对角线注意力增强
        self.pos_optimization = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            # nn.Tanh(),
            # nn.Dropout(self.hparams.dropout),
            Performer(
                dim=512,
                depth=3,
                heads=8,
                dim_head=64,
                local_window_size=128,
                kernel_fn=nn.Tanh(),
                ff_dropout=self.hparams.dropout,
                attn_dropout=self.hparams.dropout,
                shift_tokens=True,
            ),
            # nn.Tanh(),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(512, self.hparams.maxWordLen)
        )

        # 定义孪生网络分类器

        self.pre_classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 3, 512),
            nn.Dropout(self.hparams.dropout),
            nn.LeakyReLU(),
            nn.Linear(512, self.hparams.labels)
        )
        self.loss_fc = nn.CrossEntropyLoss()


        self.save_hyperparameters()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, re_spo=None):
        #         loss_fc=nn.CrossEntropyLoss(ignore_index=0)

        B, L = input_ids.size()

        x = self.model(input_ids, token_type_ids=token_type_ids.long(),
                       attention_mask=attention_mask)

        x_last_hidden_state = x.last_hidden_state

        if self.hparams.use_rnn is True:
            x_last_hidden_state, _ = self.rnn(x_last_hidden_state)
            x_last_hidden_state = self.fix(x_last_hidden_state)

        # print(x_last_hidden_state)
        # re_spo
        # print(re_spo.size())
        # for it in torch.split(re_spo, 1, dim=-1):
        #     print(it.size())
        # print(torch.split(re_spo, 1, dim=-1))
        # print(re_spo.size())
        # 拆分数据，做逐个实体对分类，关系对分类
        s, e, labels = torch.split(re_spo, 1, dim=-1)[0], torch.split(re_spo, 1, dim=-1)[2], \
                       torch.split(re_spo, 1, dim=-1)[1]
        # print(s.size(), e.size(), labels.size())
        # print(s)
        # s = rearrange(s, 'b c 1 -> b c')
        # print("all", s, e, labels.size())
        # print(s, e, labels)
        loss = 0
        items = None
        items_labels = None
        for it_s, it_e, it_l in zip(s.split(1, dim=1), e.split(1, dim=1), labels.split(1, dim=1)):
            # 自动筛选出匹配的表示
            # print("it", it_s, it_e, it_l)
            # print(it_s.sum(0))
            if torch.sum(it_s, dim=0) == 0:
                continue

            try:
                it_s_index = one_hot(it_s.view(-1).long(), num_classes=L)
                # print(it_s_index)
                # s开始位置
                mask = it_s_index == 1
                # 筛选出表示结果
                # print(x_last_hidden_state[mask])
                it_s_hidden_state = x_last_hidden_state[mask]
                # print(it_s_hidden_state.size())

                # e 开始位置
                it_e_index = one_hot(it_e.view(-1).long(), num_classes=L)
                # print(new)
                mask = it_e_index == 1
                # 筛选出表示结果
                # print(x_last_hidden_state[mask])
                it_e_hidden_state = x_last_hidden_state[mask]
                # print("it_e_hidden_state",it_e_hidden_state.size())

                # s_e=torch.cat((it_s_hidden_state,it_e_hidden_state),-1).view(B,2,-1)
                # print(s_e.size())

                emb_diff = it_e_hidden_state - it_s_hidden_state

                sim_c = torch.cat((it_e_hidden_state, it_s_hidden_state, emb_diff.abs()), -1)

                if items is None:
                    items = sim_c
                    items_labels = it_l
                else:
                    items = torch.cat((items, sim_c), 0)
                    items_labels = torch.cat((items_labels, it_l), 0)
                # print("items size:", items.size())
            except Exception as e:
                print("e", e)
                pass
        #
        if items is not None:
            # print("items", items.size())
            pooler = self.pre_classifier(items)
            # print("pooler", pooler)
            # print(it_l.view(-1))
            # print("items_labels",items_labels)
            loss1 = self.loss_fc(pooler, items_labels.view(-1).long())
            try:
                acc = accuracy(pooler.argmax(-1), items_labels.view(-1).long())
                self.log("acc", acc)
            except:
                pass
            # print("loss1",loss1)
            if loss is None:
                loss = loss1
            else:
                loss = loss + loss1
                # print(pooler.argmax(dim=-1))
                # print(it_l)

        # 关系对分类，结束
        # print("x_last_hidden_state",x_last_hidden_state)
        out_pos = self.pos_optimization(x_last_hidden_state)
        # 计算类型

        out_type = self.out_type(x_last_hidden_state)

        out_lm = self.lmhead(x_last_hidden_state)
        # print("out_pos, out_type, out_lm", out_pos, out_type, out_lm)
        return out_pos, out_type, out_lm, loss

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
