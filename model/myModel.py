# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：
一个简单的模型示例。
"""
import os
"""
MLM demo训练

"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
# from tkitAutoMask import autoMask
from torch.utils.data import DataLoader
from torchmetrics.functional import precision_recall, accuracy, f1
# from transformers import BertTokenizer


class myModel(pl.LightningModule):
    """
    EncDec
    使用transformer实现



    """

    def __init__(self, learning_rate=5e-5,
                 T_max=5,
                 optimizer_name="AdamW",
                 dropout=0.2,
                 batch_size=2,
                 trainfile="./out/train.pkt",
                 valfile="./out/val.pkt",
                 testfile="./out/test.pkt",
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 nhead=8,
                 d_model=128,
                 pretrained="uer/chinese_roberta_L-2_H-128",
                 T_mult=2,
                 T_0=500,
                 **kwargs):
        super().__init__()
        # save save_hyperparameters
        self.save_hyperparameters()
        # self.tokenizer = BertTokenizer.from_pretrained(pretrained)
        self.emb = nn.Embedding(21128, d_model, padding_idx=0)
        self.model = nn.Transformer(d_model=self.hparams.d_model,
                                    nhead=self.hparams.nhead,
                                    num_encoder_layers=self.hparams.num_encoder_layers,
                                    num_decoder_layers=self.hparams.num_decoder_layers,
                                    dropout=self.hparams.dropout,
                                    batch_first=True,

                                    )
        self.out = nn.Linear(self.hparams.d_model, 21128)
        #
        # self.tomask = autoMask(
        #     # transformer,
        #     mask_token_id=self.tokenizer.mask_token_id,  # the token id reserved for masking
        #     pad_token_id=self.tokenizer.pad_token_id,  # the token id for padding
        #     mask_prob=0.05,  # masking probability for masked language modeling
        #     replace_prob=0.90,
        #     # ~10% probability that token will not be masked, but included in loss, as detailed in the epaper
        #     mask_ignore_token_ids=[self.tokenizer.cls_token_id, self.tokenizer.eos_token_id]
        #     # other tokens to exclude from masking, include the [cls] and [sep] here
        # )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        src = self.emb(src)
        tgt = self.emb(tgt)
        # out = self.model(src, tgt, src_mask.float(), tgt_mask.float(), memory_mask, src_key_padding_mask, tgt_key_padding_mask,
        #                  memory_key_padding_mask)
        out = self.model(src, tgt)
        out = self.out(out)
        # print("out", out)
        # print("out",out.size())
        return out
        pass

    def loss_fc(self, out, src_mask, tgt, tgt_mask):
        # B,L=tgt_mask.size()
        loss_fc = nn.CrossEntropyLoss()
        active_loss = tgt_mask.view(-1) == 1
        loss = loss_fc(out.view(-1, 21128), tgt.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        src, src_mask, tgt, tgt_mask = batch
        # src, _ = self.tomask(src)

        outputs = self(src, src_mask, tgt, tgt_mask)
        loss = self.loss_fc(outputs, src_mask, tgt, tgt_mask)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward

        src, src_mask, tgt, tgt_mask = batch
        # src, _ = self.tomask(src)

        outputs = self(src, src_mask, tgt, tgt_mask)
        loss = self.loss_fc(outputs, src_mask, tgt, tgt_mask)
        metrics = {
            # "val_precision_macro": precision,
            # "val_recall_macro": recall,
            # "val_f1_macro": pred_f1,
            # "val_acc": acc,
            "val_loss": loss
        }
        # print("metrics",metrics)
        self.log_dict(metrics)
        return metrics

    def test_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        input_ids, token_type_ids, attention_mask = batch
        input_ids, labels = self.tomask(input_ids)
        outputs = self(input_ids.long(), token_type_ids.long(), attention_mask.long(), labels.long())
        pred = outputs.logits.argmax(-1)
        print("pred", pred)
        with open("test_ner.txt", "a+") as f:
            words = self.tokenizer.convert_ids_to_tokens(input_ids.view(-1).tolist())
            for i, (w, x, y, l, m) in enumerate(
                    zip(words, input_ids.view(-1).tolist(), pred.view(-1).tolist(), labels.view(-1).tolist(),
                        attention_mask.view(-1).tolist())):
                if m == 1:
                    print(w, x, y, l, m)
                    # f.write(str(y)+"--"+str(l))
                    # f.write(",".join([w,x,y,l,m]))
                    # f.write("\n")
                    # f.write("".join(words).replace("[PAD]", " "))
                    # f.write("\n")

        active_loss = attention_mask.view(-1) == 1
        precision, recall = precision_recall(pred.view(-1)[active_loss], labels.reshape(-1).long()[active_loss],
                                             average='macro', num_classes=self.hparams.num_labels)

        pred_f1 = f1(pred.view(-1)[active_loss], labels.reshape(-1).long()[active_loss], average='macro',
                     num_classes=self.hparams.num_labels)
        acc = accuracy(pred.view(-1)[active_loss], labels.reshape(-1).long()[active_loss])

        metrics = {
            "test_precision_macro": precision,
            "test_recall_macro": recall,
            "test_f1_macro": pred_f1,
            "test_acc": acc,
            "test_loss": outputs.loss
        }
        self.log_dict(metrics)
        return metrics

    def train_dataloader(self):
        train = torch.load(self.hparams.trainfile)
        return DataLoader(train, batch_size=int(self.hparams.batch_size), num_workers=24, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        val = torch.load(self.hparams.valfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=24, pin_memory=True)

    def test_dataloader(self):
        val = torch.load(self.hparams.testfile)
        return DataLoader(val, batch_size=int(self.hparams.batch_size), num_workers=24, pin_memory=True)

    def configure_optimizers(self):
        """优化器 自动优化器"""
        optimizer = getattr(optim, self.hparams.optimizer_name)(self.parameters(), lr=self.hparams.learning_rate)


        #         使用自适应调整模型
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=500000, factor=0.8,
        #                                                        verbose=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams.T_0, T_mult=self.hparams.T_mult, eta_min=0, last_epoch=-1, verbose=False)
        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'name': "lr_scheduler",
            'monitor': 'train_loss',  # 监听数据变化
            'strict': True,
        }
        #         return [optimizer], [lr_scheduler]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


if __name__ == '__main__':
    pass
