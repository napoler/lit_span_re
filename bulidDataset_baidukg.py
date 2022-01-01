# -*- coding: utf-8 -*-
"""
作者：　terrychan
Blog: https://terrychan.org
# 说明：

"""
import csv
import json
import os
import pickle

import pkuseg
import torch
from torch.utils.data import TensorDataset
# from performer_pytorch import Performer
# from reformer_pytorch import Reformer
from tqdm.auto import tqdm
from transformers import BertTokenizerFast
import random
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from tkitAutoTokenizerPosition import AutoTokenizerPosition, autoBIO, autoSpan, AutoPos
from tkitDatasetEx.AutoClear import AutoClear

# from tkitDatasetEx.readData import  readCMeEE

# from sklearn.preprocessing import MultiLabelBinarizer
csv.field_size_limit(100000000000)

"""


128长度
基于逐个字符处理
包含了span的位置信息
span类型

使用BertTokenizerFast将位置映射操作

new_input_ids, # 文本编码
new_token_type_ids, # 文本类型编码
new_attention_mask, #注意力编码

new_tagtype, # 实体类型
new_tag, # 实体位置信息





"""

datapath = "data/dataset"
outPath = "data/char_bert_ner"
DEBUG = False  # 设置测试限制20条数据
MaxRE=10
try:
    os.mkdir(outPath)
except:
    pass
# seg = pkuseg.pkuseg(model_name='medicine', postag=False,
#                     user_dict="./mydict.txt")  # 程序会自动下载所对应的细领域模型
# seg = pkuseg.pkuseg(postag=False, user_dict="./mydict.txt")  # 程序会自动下载所对应的细领域模型
tokenizer = BertTokenizerFast.from_pretrained("uer/chinese_roberta_L-2_H-128", return_offsets_mapping=True,
                                              model_max_length=1000000,
                                              do_basic_tokenize=False
                                              #   tokenize_chinese_chars=True
                                              )

print("tokenizer", tokenizer)

# 初始化自动修正
apos = AutoClear(tokenizer=tokenizer)


def readData(datapath):
    """
    读取CMeEE数据，
    返回fileName, datajson
    :param datapath:
    :return fileName, datajson:
    """
    fileList = ["train_data.json", "dev_data.json"]

    for fileName in fileList:
        datajson = []
        file = os.path.join(datapath, fileName)
        print("file", file)
        with open(file, "r") as f:
            for line in f:
                # print(json.loads(line))
                datajson.append(json.loads(line))
            yield fileName, datajson
        pass


# if DEBUG:
#     datajson = datajson[:20]
kglabels = {"O": 0, }
relabels = {"O": 0, }
for fileName, datajson in readData(datapath):
    # print("datajson", datajson)
    # print("fileName", fileName)
    if fileName == "train_data.json":
        for it in datajson:
            for one in it['spo_list']:
                wtype = one["object_type"]

                try:
                    kglabels[wtype] += 1
                except:
                    kglabels[wtype] = 1

                wtype = one["subject_type"]
                try:
                    kglabels[wtype] += 1
                except:
                    kglabels[wtype] = 1

                try:
                    relabels[one["predicate"]] += 1
                except:
                    relabels[one["predicate"]] = 1

    pass

print("kglabels", kglabels)


def save(datajson, save_filename="train.pkt", save_labels=False, fakeNum=1, maxLen=2048, maxWordLen=32, model_len=512,
         MaxRE=10):
    # 最大单词长度
    print("datajson", len(datajson))

    kglabels_list = list(kglabels.keys())
    relabels_list = list(relabels.keys())
    # relabels={"非实体":0,"无关系":0}
    # relabels = {"无关系": 0}
    # relabels_list = list(relabels.keys())
    datas = {"texts": [], "tags": [], "tagtype": [],
             "token_type_ids_head": [], "reType": [], "reSpo": []}

    total_re = 0
    total_re_bad = 0

    # 处理分词方案数据1
    csvfilebase = open(save_filename + "base.csv", 'w')
    writer_base = csv.writer(csvfilebase)
    with open(save_filename + ".csv", 'w') as csvfile:
        writer = csv.writer(csvfile)
        for i, it in tqdm(enumerate(datajson)):

            text = it['text']
            or_text = apos.clearText(text)

            for ii in range(fakeNum):
                # for ii in range(1):
                # 构建伪造数据
                rand_num = random.randint(0, 12)
                # rand_num=0
                WordList = list(or_text)
                WordList = rand_num * ["ر"] + WordList

                tag = (len(WordList) + 1) * [0]
                # 词性
                tagtype = (len(WordList) + 1) * [0]

                reSpo = MaxRE*int(maxLen/model_len) * [[0, 0, 0]]

                # 使用-100计算交叉商忽略
                # rePo = np.full((maxLen, model_len), 0)
                # 构建矩阵
                # token_type = np.full((maxLen, model_len), 0)
                # print("rePo1",rePo)
                good = False
                wordStartIndex = {}
                if it['spo_list']:
                    for one in it['spo_list']:
                        for key in one.keys():
                            if key in ["object", "subject"]:
                                # print(key)

                                word = one[key]
                                wtype = one[key + "_type"]

                                starPos = it['text'].find(word)
                                endPos = starPos + len(word)

                                # print("text",it['text'])
                                # print("starPos",word,starPos,endPos)
                                # 加入偏移
                                endPos = endPos + rand_num
                                starPos = starPos + rand_num

                                if starPos > maxLen-5 or endPos > maxLen-5:
                                    continue
                                # wordStartIndex[one['id']] = starPos
                                # writer_test.writerow(["".join(WordList[s_start:s_start+wordLen])])
                                # print(WordList[s_start:s_end])
                                # index=(starPos+1)%model_len

                                index = (starPos + 1) % model_len
                                endindex = (endPos + 1) % model_len
                                # 修改成差值，，也就是计算向后推理多少个词语

                                writer_base.writerow([wtype, word, "".join(WordList[starPos:endPos])])
                                try:
                                    if (endPos - starPos) < maxWordLen and starPos < maxLen - 10:
                                        # 相对位置
                                        tag[starPos + 1] = endPos - starPos
                                        # 类型
                                        tagtype[starPos + 1] = kglabels_list.index(wtype)
                                        # print("index",index)
                                        # for index_i in range(index, endindex):
                                        #     token_type[starPos+1][index_i] = 1
                                        good = True

                                        pass
                                except Exception as e:
                                    print("e", e)
                                    pass

                if it['spo_list']:
                    for spo_i, one in enumerate(it['spo_list']):

                        spo = {}
                        for key in one.keys():
                            if key in ["object", "subject"]:
                                # print(key)

                                word = one[key]
                                wtype = one[key + "_type"]

                                starPos = it['text'].find(word)
                                endPos = starPos + len(word)


                                # print("text",it['text'])

                                # 加入偏移
                                endPos = endPos + rand_num
                                starPos = starPos + rand_num

                                if starPos >maxLen-5 or endPos >maxLen-5:
                                    continue
                                spo[key] = endPos
                                spo["predicate"] = relabels_list.index(one["predicate"])

                        # print("spo", list(spo.values()))
                        try:
                            if len(list(spo.values()))==3:
                                reSpo[spo_i] = list(spo.values())
                        except:
                            pass
                if good != True:
                    print("pass")
                    continue
                # print("WordList",WordList)
                # 修正文字位置
                # text = " ".join(WordList)
                WordList = apos.clearTextDec(WordList)
                text = " ".join(WordList)
                # print(text[:100])
                datas['texts'].append(text)
                nt = tag + [0] * maxLen
                tag = nt[:maxLen]
                datas["tags"].append(tag)
                nt = tagtype + [0] * maxLen
                tagtype = nt[:maxLen]
                datas["tagtype"].append(tagtype)

                datas["reSpo"].append(reSpo)

                # print(WordList)
            # break
            for iit in zip(WordList, tag[1:], tagtype[1:]):
                # print(iit)
                writer.writerow(iit)

    csvfilebase.close()
    if save_labels:
        pickle.dump(kglabels_list, open(os.path.join(outPath, "kglabels_list.p"), "wb"))

    # print("datas['texts']",datas['texts'][:5])
    # 生成初期矩阵数据
    textTensor = tokenizer(datas['texts'], padding="max_length", max_length=maxLen, truncation=True,
                           return_tensors="pt")

    # 起止位置相对
    tagsTensor = torch.Tensor(datas["tags"])
    tagtypeTensor = torch.Tensor(datas["tagtype"])

    print("spo",len(datas["reSpo"]))
    reSpoTensor = torch.Tensor(datas["reSpo"])

    # print(reSpoTensor.select(1, 0))

    # # 头实体矩阵
    # token_type_ids_head = torch.Tensor(datas["token_type_ids_head"])
    # # 关系矩阵
    # rePoTensor = torch.Tensor(datas["rePo"])

    myDataset = TensorDataset(textTensor["input_ids"].view(-1, model_len),
                              textTensor["token_type_ids"].view(-1, model_len),
                              textTensor["attention_mask"].view(-1, model_len),
                              tagsTensor.view(-1, model_len),
                              tagtypeTensor.view(-1, model_len),
                              reSpoTensor.view(-1, MaxRE,3),

                              )

    fl = len(myDataset)

    print("总长度：", fl)
    print("kglabels_list", len(kglabels_list), kglabels_list)
    torch.save(myDataset, save_filename)


for fileName, datajson in readData(datapath):
    # print("datajson", datajson)
    # print("fileName", fileName)
    datajson = datajson[:5000]
    save(datajson, save_filename=os.path.join(outPath, fileName + ".pkt"), save_labels=True,MaxRE=MaxRE,maxLen=128,model_len=128, fakeNum=5)

if __name__ == '__main__':
    pass
