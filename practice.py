'''
import numpy as np
x=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
for i in range(0,len(x)):
    print("Table\n\n")
    for row in range(0,x[i].shape[0]):
        print("\n")
        for column in range(0,x[i].shape[1]):
            print(x[i][row][column])
            print("\t")
'''

'''
from typing import Sequence
from typing import Tuple
from typing import List
from typing import Union
from typing import overload
def fun(a: int, b: int) -> int:
    v : List[str]= []
    return a+b
'''
'''
some time the functions can take or handle more than one type that time we use :
1) Union : we have to specify all the types including none
2) Optional : It will by default take None
'''
'''
def f(a : Union[int ,float],b : Union[int , float]) -> Union[int , float] :
    return a-b

'''
'''
Function OverLoad
'''
'''
@overload
def ff(a : Union[int ,float],b : Union[int,float],c : Union[int,float])-> Union[int,float]:
    return a+b-c

@overload
def ff(st : str)-> str:
    return st

'''
'''
keeping the all datatypes into one variable
'''
'''
from typing import TypeVar
Anystr = TypeVar('Anystr',str,bytes)

def hg(st : Anystr)->Anystr:
    return st
'''
'''
int
float
str
bytes
bool
'''

'''
if dont know the type the keep Any type ,the under Any all datatype will come
'''
'''
from typing import Any
def jk(st : Any) -> Any:
    return st

'''
'''
documneting the class objects
'''

#from typing import Protocol
from typing import Union
#from typing import TypeVar
#T = TypeVar('T', bound='A')
'''
class A:
    def __init__(self,r : A):
        self.r = r

    def __add__(self, other: A) -> A:
        return A(self.r + other.r)

    def fun(self)->A:
        return self.r

def pp(obj : A)->A:
    return obj.fun()

'''
import os
import json
import torch
import random
import regex
import numpy
from difflib import SequenceMatcher
from string import ascii_uppercase, digits, punctuation


#type hinting libraries
from os import DirEntry
from typing import Tuple
from typing import List

import os
import numpy as np
import cv2
import torch
from torch import optim,nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import json
from bilstm_utils import VOCAB, robust_padding, pred_to_dict, create_data, create_test_data
from ctpn_utils import cal_rpn
from ctpn import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss

#paths
icdar_img_dir = r'./Uploads/receipts/images'
icdar_ant_dir = r'./Uploads/receipts/annotations'
icdar_keys_dir = r'./Uploads/receipts/key_info'

# Model Configuration
num_workers = 0
anchor_scale = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7

RPN_POSITIVE_NUM = 150
RPN_TOTAL_NUM = 300

IMAGE_MEAN = [123.68, 116.779, 103.939]
OHEM = True

class ICDARDataset(Dataset):
    """
    Dataset class for preparing data to suit the model's input requirements
    """

    def __init__(self ,datadir :str,labelsdir :str) -> None:
        '''
        :param datadir: image's directory
        :param labelsdir: annotations' directory
        '''
        if not os.path.isdir(datadir):
            raise Exception('[ERROR] {} is not a directory'.format(datadir))
        if not os.path.isdir(labelsdir):
            raise Exception('[ERROR] {} is not a directory'.format(labelsdir))

        self.datadir = datadir
        self.img_names = os.listdir(self.datadir)
        self.labelsdir = labelsdir

    def __len__(self):
        return len(self.img_names)
