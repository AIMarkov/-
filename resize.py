#图像预处理
from PIL import Image
import os
import re
path="data/images/"
filelist=os.listdir("data/images")
Wideth=[]
Height=[]
for Id in filelist:
    filepath=""
    filepath=path+Id
    im=Image.open(filepath,'r')
    Size=im.size
    Wideth.append(Size[0])
    Height.append(Size[1])
minWideth=min(Wideth)
minHeight=min(Height)

savepath="processed/"

filelist_no_JPG=[]
for Id in filelist:
    filepath=""
    filepath=path+Id
    im=Image.open(filepath)
    imresize=im.resize((256,256))
    imresize=imresize.convert('L')#若不想用彩色的，则可以直接把这句删除
    #imresize.show()
    #print(Sfilepath)
    #subSfilepath=re.sub('JPG',"jpg",Sfilepath)
    #print(subSfilepath)
    Id = re.sub('.JPG', '', Id)
    Id = re.sub('.jpg', '', Id)
    Sfilepath = ""
    Sfilepath = savepath + Id
    filelist_no_JPG.append(Id)
    imresize.save(Sfilepath,'jpeg')



