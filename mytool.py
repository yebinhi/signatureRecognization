import os

path = '/Users/binye/MyWorkspace/signature-verification/data/english/本人签名+非本人签名/all/'
# os.chdir('/Users/binye/MyWorkspace/signature-verification/data/english/本人签名+非本人签名/all/')
# print(os.getcwd())

# for count, f in enumerate(os.listdir()):
#     new = f.replace('072', '073')
#     new_1 = new.replace('-', '_')
#     print(new_1)
#
#     #new_name = f'{f_name}{f_ext}'
#     os.rename(f, new_1)
items=['070','071','072','073','074']

for item in items:
    f_name = item
    f_forge_name = item+'_forge'

    new_path = path+f_name
    new_path_forge = path+f_forge_name
    os.chdir(new_path)
    list_ = os.listdir(os.chdir(new_path))
    list_forge = os.listdir(os.chdir(new_path_forge))
    for i in range(len(list_)):
        j= i
        for j in range(len(list_)):
            print(f_name+'/'+list_[i]+','+f_name+'/'+list_[j]+',1')

    for i in range(len(list_)):
        j=0
        for j in range(len(list_forge)):
            print(f_name+'/'+list_[i]+','+f_forge_name+'/'+list_forge[j]+',0')




