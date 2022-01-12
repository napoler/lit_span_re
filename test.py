# import torch
# from einops import rearrange, reduce, repeat
#
# s = torch.tensor([[[11.],
#                    [24.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.]],
#                   [[25.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.],
#                    [0.]]])
#
# print(s)
# # s=rearrange(s, 'b c 1 -> b c')
# x = torch.randn(2, 64, 8)
# print(x)
# out = torch.gather(x, 1, s.long())
# print(out.size())
#
# indices = torch.tensor([0, 1])
# out=torch.index_select(x, 1, indices)
# print(out.size())
# print(out)
# # torch.index_select(x, 1, indices)
#
#
# print("进行逐个操作")
# for it,xx in zip(s.split(1, dim=0),x.split(1,dim=0)):
#     print(it)
#     print(xx)
# # 有效测试
# # https://colab.research.google.com/drive/1XPvu3RiykluNatK3grQRV_6XS3BAW8BB#scrollTo=Gx1pRjWrxnTz&line=1&uniqifier=1
#
#
#
#
#
#
# from torch.nn.functional import one_hot
# for it in s.split(1, dim=1):
#   print(it.squeeze(-1))
#   index=it.squeeze(-1)
#   print("index.view(-1)",index.view(-1))
#   new=one_hot(index.view(-1).long(),num_classes=64)
#   print(new)
#   mask= new==1
#   print(x[mask])
a={'O': 0, '人物': 292747, '影视作品': 100172, '目': 10342, '生物': 10342, 'Number': 5272, 'Date': 39716, '国家': 12781, '网站': 12853, '网络小说': 13103, '图书作品': 33824, '歌曲': 54612, '地点': 24150, '气候': 920, '行政区': 7713, '学校': 11510, '企业': 19385, '出版社': 17686, '书籍': 17686, '音乐专辑': 10398, '城市': 972, '景点': 575, '电视综艺': 2879, '机构': 5307, '作品': 553, '语言': 144, '学科专业': 46}
print(len(a))
