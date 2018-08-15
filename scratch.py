# -*- coding: UTF-8 -*-
import xlrd
import re
from datetime import datetime
from xlrd import xldate_as_tuple
import json
import shutil


def read_excel(path1):
    workbook=xlrd.open_workbook(path1)
    sheet0=workbook.sheet_by_index(0)
    return sheet0.cell(11,13).value

# def str_change(str,index):
#     str=list(str)
#     str[]
#

def add_zero(str,index):
    str=list(str)
    str[index]='0'+str[index]
    str=''.join(str)
    return str

def add_slash(str):
    str=list(str)
    str[4]='/'+str[4]
    str[6]='/'+str[6]
    str = ''.join(str)
    return str


def zero_ornot(str):
    pos1 = str.index(u'年')
    pos2 = str.index(u'月')
    interval=str[pos1+1:pos2]
    if len(interval)==1:
        return 1
    else:
        return 0

def add_quote(str):
    str=list(str)
    str[0]='"'+str[0]
    str[len(str)-1]=str[len(str)-1]+'"'
    str = ''.join(str)
    return str

def add_zhuan(str):
    str = list(str)
    str[0] = 'u"' + str[0]
    str[len(str) - 1] = str[len(str) - 1] + '"'
    str = ''.join(str)
    return str

if __name__ == '__main__':
    path = raw_input("请输入文件目录:")
    path_final=raw_input("请输入结果文件夹:")
    path=path.decode('utf-8')

    aa={'0':13,'1':13,'2':11,'5':11}
    b={'0':12,'1':12,'2':11,'5':11}
    article_path = {'0': path_final+"\\"+"鑫益.json", '1': path_final+"\\"+"鑫益展期.json",
                    '2': path_final+"\\"+"晟益.json", '5': path_final+"\\"+"益先锋.json"}
    #article_path={'0':"D:\\data_store_temp\\鑫益.json",'1': "D:\\data_store_temp\\鑫益展期.json",'2': "D:\\data_store_temp\\晟益.json",'5': "D:\\data_store_temp\\益先锋.json"}
    # wbk = xlwt.Workbook()
    for l in  [0,1,2,5]:
        workbook=xlrd.open_workbook(path)
        sheet0=workbook.sheet_by_index(l)
        nrow=sheet0.nrows-b[str(l)]+1
        mark=0
        # sheet = wbk.add_sheet('sheet' + str(l))
        for j in range(nrow):
            a=sheet0.cell(b[str(l)]+j-1,aa[str(l)]).value
            name=sheet0.cell(b[str(l)]+j-1,0).value.encode("utf-8")
            cell=sheet0.cell_value(b[str(l)]+j-1,4)
            date=datetime(*xldate_as_tuple(cell, 0))
            cell = date.strftime('%Y/%m/%d')
            final_observation=cell
            a_split=a.split()
            cc=[]
            result_path=path_final+"\\"+name+".json"
            print result_path
            start_path=article_path[str(l)].decode("utf-8")
            result_path=result_path.decode("utf-8")
            shutil.copy(start_path, result_path)
        # shishi=a_split[1]
        # tmp="2"
        # npos=shishi.index(tmp)
        # haha=shishi[npos+2:len(shishi)]
        # pos1=haha.index(u'年')
        # pos2=haha.index(u'月')
        # print haha[pos1+1:pos2]
            for i in range(len(a_split)):
                tmp=str(i+1)
                shishi=a_split[i]
                npos=shishi.index(tmp)
                if i<=8:
                    xixi= shishi[npos + 2:len(shishi)]
                    num=zero_ornot(xixi)
                    xixi= str(re.sub("\D", "", xixi))
                    if num:
                        xixi=add_zero(xixi,4)
                    if len(xixi)<8:
                        xixi=add_zero(xixi,6)
                    xixi=add_slash(xixi)
                    # xixi=add_quote(xixi)
                    cc.append(xixi)

                else:
                    xixi= shishi[npos + 3:len(shishi)]
                    num = zero_ornot(xixi)
                    xixi = re.sub("\D", "", xixi)
                    if num:
                        xixi = add_zero(xixi, 4)
                    if len(xixi) < 8:
                        xixi = add_zero(xixi, 6)
                    xixi=add_slash(xixi)
                    # xixi = add_quote(xixi)
                    cc.append(xixi)
            print final_observation
            cc.append(final_observation)
            print cc
            f=open(result_path)
            t=json.load(f)
            f=open(result_path,"w")
            t['KOSchedule'] = cc
            f.write(json.dumps(t, ensure_ascii=False))
            f.close()

            # sheet.write(j,0,name)
            # sheet.write(j,1,cc)
    # wbk.save("D:\\result_0814.xls")


