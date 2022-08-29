import os

# path_dir = "./dataset_QXY3/audio/data_wav"
# s_list = []
# for p in os.listdir(path_dir):
#     path_dir_1 = os.path.join(path_dir, p)
#     # print(path_dir_)
#     for file in os.listdir(path_dir_1):
#         # print(file)
#         s = os.path.join(path_dir_1, file)
#         s = s + '\t' + file.split('_')[1]
#         s = s[1:]
#         # print(s)
#         s_list.append(s)
#         # break
#
# print(len(s_list))
# with open("./dataset_QXY3/annotation/Qxueyou.txt", 'w', encoding='utf-8') as fp:
#     for i in s_list:
#         fp.write(i)
#         fp.write('\n')



path_dir = "./dataset_QXY3/audio/data_wav/val"
s_list = []

for file in os.listdir(path_dir):
    # print(file)
    s = os.path.join(path_dir, file)
    s = s + ',' + file.split('_')[2]
    s = s[2:]
    # print(s)
    s_list.append(s)
    # break

# print(len(s_list))
with open("./dataset_QXY3/val.index", 'w+', encoding='utf-8') as fp:
    for i in s_list:
        fp.write(i)
        fp.write('\n')