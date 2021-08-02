import pdb

import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import json
import numpy as np
import pickle
import os
import torch
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
import argparse
from vncorenlp import VnCoreNLP
import re
import string

VN_CHARS_LOWER = u'ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđð'
VN_CHARS_UPPER = u'ẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸÐĐ'
VN_CHARS = VN_CHARS_LOWER + VN_CHARS_UPPER

replace_list = {
        'òa': 'oà', 'óa': 'oá', 'ỏa': 'oả', 'õa': 'oã', 'ọa': 'oạ', 'òe': 'oè', 'óe': 'oé', 'ỏe': 'oẻ',
        'õe': 'oẽ', 'ọe': 'oẹ', 'ùy': 'uỳ', 'úy': 'uý', 'ủy': 'uỷ', 'ũy': 'uỹ', 'ụy': 'uỵ', 'uả': 'ủa',
        'ả': 'ả', 'ố': 'ố', 'u´': 'ố', 'ỗ': 'ỗ', 'ồ': 'ồ', 'ổ': 'ổ', 'ấ': 'ấ', 'ẫ': 'ẫ', 'ẩ': 'ẩ',
        'ầ': 'ầ', 'ỏ': 'ỏ', 'ề': 'ề', 'ễ': 'ễ', 'ắ': 'ắ', 'ủ': 'ủ', 'ế': 'ế', 'ở': 'ở', 'ỉ': 'ỉ',
        'ẻ': 'ẻ', 'àk': u' à ', 'aˋ': 'à', 'iˋ': 'ì', 'ă´': 'ắ', 'ử': 'ử', 'e˜': 'ẽ', 'y˜': 'ỹ', 'a´': 'á',
        # Quy các icon về 2 loại emoj: Tích cực hoặc tiêu cực
        "👹": "tệ", "👻": "tốt", "💃": "tốt", '🤙': ' tốt ', '👍': ' tốt ',
        "💄": "tốt", "💎": "tốt", "💩": "tốt", "😕": "tệ", "😱": "tệ", "😸": "tốt",
        "😾": "tệ", "🚫": "tệ", "🤬": "tệ", "🧚": "tốt", "🧡": "tốt", '🐶': ' tốt ',
        '👎': ' tệ ', '😣': ' tệ ', '✨': ' tốt ', '❣': ' tốt ', '☀': ' tốt ',
        '♥': ' tốt ', '🤩': ' tốt ', 'like': ' tốt ', '💌': ' tốt ',
        '🤣': ' tốt ', '🖤': ' tốt ', '🤤': ' tốt ', ':(': ' tệ ', '😢': ' tệ ',
        '❤': ' tốt ', '😍': ' tốt ', '😘': ' tốt ', '😪': ' tệ ', '😊': ' tốt ',
        '?': ' ? ', '😁': ' tốt ', '💖': ' tốt ', '😟': ' tệ ', '😭': ' tệ ',
        '💯': ' tốt ', '💗': ' tốt ', '♡': ' tốt ', '💜': ' tốt ', '🤗': ' tốt ',
        '^^': ' tốt ', '😨': ' tệ ', '☺': ' tốt ', '💋': ' tốt ', '👌': ' tốt ',
        '😖': ' tệ ', '😀': ' tốt ', ':((': ' tệ ', '😡': ' tệ ', '😠': ' tệ ',
        '😒': ' tệ ', '🙂': ' tốt ', '😏': ' tệ ', '😝': ' tốt ', '😄': ' tốt ',
        '😙': ' tốt ', '😤': ' tệ ', '😎': ' tốt ', '😆': ' tốt ', '💚': ' tốt ',
        '✌': ' tốt ', '💕': ' tốt ', '😞': ' tệ ', '😓': ' tệ ', '️🆗️': ' tốt ',
        '😉': ' tốt ', '😂': ' tốt ', ':v': '  tốt ', '=))': '  tốt ', '😋': ' tốt ',
        '💓': ' tốt ', '😐': ' tệ ', ':3': ' tốt ', '😫': ' tệ ', '😥': ' tệ ',
        '😃': ' tốt ', '😬': ' 😬 ', '😌': ' 😌 ', '💛': ' tốt ', '🤝': ' tốt ', '🎈': ' tốt ',
        '😗': ' tốt ', '🤔': ' tệ ', '😑': ' tệ ', '🔥': ' tệ ', '🙏': ' tệ ',
        '🆗': ' tốt ', '😻': ' tốt ', '💙': ' tốt ', '💟': ' tốt ',
        '😚': ' tốt ', '❌': ' tệ ', '👏': ' tốt ', ';)': ' tốt ', '<3': ' tốt ',
        '🌝': ' tốt ', '🌷': ' tốt ', '🌸': ' tốt ', '🌺': ' tốt ',
        '🌼': ' tốt ', '🍓': ' tốt ', '🐅': ' tốt ', '🐾': ' tốt ', '👉': ' tốt ',
        '💐': ' tốt ', '💞': ' tốt ', '💥': ' tốt ', '💪': ' tốt ',
        '💰': ' tốt ', '😇': ' tốt ', '😛': ' tốt ', '😜': ' tốt ',
        '🙃': ' tốt ', '🤑': ' tốt ', '🤪': ' tốt ', '☹': ' tệ ', '💀': ' tệ ',
        '😔': ' tệ ', '😧': ' tệ ', '😩': ' tệ ', '😰': ' tệ ', '😳': ' tệ ',
        '😵': ' tệ ', '😶': ' tệ ', '🙁': ' tệ ',
        # Chuẩn hóa 1 số sentiment words/English words
        ':))': '  tốt ', ':)': ' tốt ', 'ô kêi': ' ok ', 'okie': ' ok ', ' o kê ': ' ok ',
        'okey': ' ok ', 'ôkê': ' ok ', 'oki': ' ok ', ' oke ': ' ok ', ' okay': ' ok ', 'okê': ' ok ',
        ' tks ': u' cám ơn ', 'thks': u' cám ơn ', 'thanks': u' cám ơn ', 'ths': u' cám ơn ', 'thank': u' cám ơn ',
        '⭐': 'star ', '*': 'star ', '🌟': 'star ', '🎉': u' tốt ',
        'kg ': u' không ', 'not': u' không ', u' kg ': u' không ', '"k ': u' không ', ' kh ': u' không ',
        'kô': u' không ', 'hok': u' không ', ' kp ': u' không phải ', u' kô ': u' không ', '"ko ': u' không ',
        u' ko ': u' không ', u' k ': u' không ', 'khong': u' không ', u' hok ': u' không ',
        'he he': ' tốt ', 'hehe': ' tốt ', 'hihi': ' tốt ', 'haha': ' tốt ', 'hjhj': ' tốt ',
        ' lol ': ' tệ ', ' cc ': ' tệ ', 'cute': u' dễ thương ', 'huhu': ' tệ ', ' vs ': u' với ',
        'wa': ' quá ', 'wá': u' quá', 'j': u' gì ', '“': ' ',
        ' sz ': u' cỡ ', 'size': u' cỡ ', u' đx ': u' được ', 'dk': u' được ', 'dc': u' được ', 'đk': u' được ',
        'đc': u' được ', 'authentic': u' chuẩn chính hãng ', u' aut ': u' chuẩn chính hãng ',
        u' auth ': u' chuẩn chính hãng ', 'thick': u' tốt ', 'store': u' cửa hàng ',
        'shop': u' cửa hàng ', 'sp': u' sản phẩm ', 'gud': u' tốt ', 'god': u' tốt ', 'wel done': ' tốt ',
        'good': u' tốt ', 'gút': u' tốt ',
        'sấu': u' xấu ', 'gut': u' tốt ', u' tot ': u' tốt ', u' nice ': u' tốt ', 'perfect': 'rất tốt',
        'bt': u' bình thường ',
        'time': u' thời gian ', 'qá': u' quá ', u' ship ': u' giao hàng ', u' m ': u' mình ', u' mik ': u' mình ',
        'ể': 'ể', 'product': 'sản phẩm', 'quality': 'chất lượng', 'chat': ' chất ', 'excelent': 'hoàn hảo',
        'bad': 'tệ', 'fresh': ' tươi ', 'sad': ' tệ ',
        'date': u' hạn sử dụng ', 'hsd': u' hạn sử dụng ', 'quickly': u' nhanh ', 'quick': u' nhanh ',
        'fast': u' nhanh ', 'delivery': u' giao hàng ', u' síp ': u' giao hàng ',
        'beautiful': u' đẹp tuyệt vời ', u' tl ': u' trả lời ', u' r ': u' rồi ', u' shopE ': u' cửa hàng ',
        u' order ': u' đặt hàng ',
        'chất lg': u' chất lượng ', u' sd ': u' sử dụng ', u' dt ': u' điện thoại ', u' nt ': u' nhắn tin ',
        u' tl ': u' trả lời ', u' sài ': u' xài ', u'bjo': u' bao giờ ',
        'thik': u' thích ', u' sop ': u' cửa hàng ', ' fb ': ' facebook ', ' face ': ' facebook ', ' very ': u' rất ',
        u'quả ng ': u' quảng  ',
        'dep': u' đẹp ', u' xau ': u' xấu ', 'delicious': u' ngon ', u'hàg': u' hàng ', u'qủa': u' quả ',
        'iu': u' yêu ', 'fake': u' giả mạo ', 'trl': 'trả lời', '><': u' tốt ',
        ' por ': u' tệ ', ' poor ': u' tệ ', 'ib': u' nhắn tin ', 'rep': u' trả lời ', u'fback': ' feedback ',
        'fedback': ' feedback ',
        # dưới 3* quy về 1*, trên 3* quy về 5*
        '6 sao': ' 5 star ', '6 star': ' 5 star', '5star': ' 5star ', '5 sao': ' 5star ', '5sao': ' 5star ',
        'starstarstarstarstar': ' 5star ', '1 sao': ' 1star ', '1sao': ' 1star ', '2 sao': ' 1star ', '2sao': ' 1star ',
        '2 starstar': ' 1star ', '1star': ' 1star ', '0 sao': ' 1star ', '0star': ' 1star ', }

def no_marks(s):
    __INTAB = [ch for ch in VN_CHARS]
    __OUTTAB = "a"*17 + "o"*17 + "e"*11 + "u"*11 + "i"*5 + "y"*5 + "d"*2
    __OUTTAB += "A"*17 + "O"*17 + "E"*11 + "U"*11 + "I"*5 + "Y"*5 + "D"*2
    __r = re.compile("|".join(__INTAB))
    __replaces_dict = dict(zip(__INTAB, __OUTTAB))
    result = __r.sub(lambda m: __replaces_dict[m.group(0)], s)
    return result


def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1E0-\U0001F1FF"
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def format_line(line):
    line = line.replace("\n","").strip()
    line = line.replace(u"\ufeff","") #"\ufeff"
    line = line.replace("\\", " ")

    line = re.sub(r'([A-Z])\1+', lambda m: m.group(1).upper(), line, flags=re.IGNORECASE)

    # Chuyển thành chữ thường
    line = line.lower()

    # Chuẩn hóa tiếng Việt, xử lý emoj, chuẩn hóa tiếng Anh, thuật ngữ


    for k, v in replace_list.items():
        line = line.replace(k, v)

    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    line = line.translate(translator)

    line = deEmojify(line)
    line = re.sub("\s+", " ", line)
    return line

def convert_lines(df, vocab, bpe, max_sequence_length):
    outputs = np.zeros((len(df), max_sequence_length))
    
    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        #pdb.set_trace()
        subwords = bpe.encode('<s> '+row.text+' </s>')
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length] 
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
        outputs[idx,:] = np.array(input_ids)
    return outputs


def convert_lines_cnn(df, vocab, bpe, max_sequence_length, bert_leng = 256):
    outputs = np.zeros((len(df), max_sequence_length))

    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # pdb.set_trace()
        subwords = bpe.encode('<s> ' + format_line(row.text) + ' </s>')
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        if len(input_ids) > bert_leng:
            input_ids[256 - 1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ] * (max_sequence_length - len(input_ids))

        if( len(input_ids) < max_sequence_length):
            input_ids = input_ids +[pad_id, ] * (max_sequence_length - len(input_ids))

        #print(len(input_ids))
        outputs[idx, :] = np.array(input_ids)
    return outputs



def convert_lines_with_Viet_setiwordnet(df, vocab, bpe, max_sequence_length):

    outputs = np.zeros((len(df), max_sequence_length))
    """
    pathToSentiFile = "./VietSentiWordnet/VietSentiWordnet_Ver1.3.5.txt"

    load setiwordnetFile

    Senti = dict({})
    f = open(pathToSentiFile, 'r', encoding="utf-8")
    first = True
    for i in f:
        if i.strip() == "":
            continue

        if (first):
            first = False
            continue

        ws = i.split('\t')
        key = ws[4].replace("#", "")
        for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            key = key.replace(i, "")
        # print(ws[2])
        pos = float(ws[2])
        neg = float(ws[3])
        if key not in Senti:
            Senti.update({key: [pos, neg]})
    f.close()
    """
    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        #pdb.set_trace()
        value = " tích_cực"
        if(row.label == 1):
            value = " tiêu_cực"

        subwords = bpe.encode('<s> ' + row.text + value + ' </s>')
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()

        if len(input_ids) > max_sequence_length:
            input_ids = input_ids[:max_sequence_length]
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ] * (max_sequence_length - len(input_ids))

        outputs[idx, :] = np.array(input_ids)
    return outputs

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def GetVec (SentiDirtional ,string):
    global Senti
    pos, neg, c=0,0, 0
    for w in string.split():
        if w in Senti:
            pos = pos+ Senti[w][0]
            neg = pos+ Senti[w][1]
    pos= pos/len(string)  # pos= pos/len(s) pos= pos/c
    neg= neg/len(string)  #neg = neg/len(s) neg= neg/c
    return pos, neg

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
"""
if __name__ == '__main__':
    # Load the dictionary
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--train_path', type=str, default='./data/train.csv')
    parser.add_argument('--dict_path', type=str, default="./phobert/dict.txt")
    parser.add_argument('--config_path', type=str, default="./phobert/config.json")
    parser.add_argument('--rdrsegmenter_path', type=str, required=True)
    parser.add_argument('--pretrained_path', type=str, default='./phobert/model.bin')
    parser.add_argument('--max_sequence_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--accumulation_steps', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=69)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--ckpt_path', type=str, default='./models')
    parser.add_argument('--bpe-codes', default="./phobert/bpe.codes", type=str, help='path to fastBPE BPE')

    args = parser.parse_args()
    bpe = fastBPE(args)
    rdrsegmenter = VnCoreNLP(args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m')

    seed_everything(69)

    vocab = Dictionary()
    vocab.add_from_file(args.dict_path)

    # Load training data
    train_df = pd.read_csv(args.train_path, sep='\t').fillna("###")
    
    Chuẩn hóa lại chuỗi dùng tokenize sản phẩm -> sản_phẩm 
    
    train_df.text = train_df.text.progress_apply(
        lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.tokenize(x)]))
    y = train_df.label.values
    
    Trả về ma trận vector embedding [dòng, 256]
    
    X_train = convert_lines_with_Viet_setiwordnet(train_df, vocab, bpe, args.max_sequence_length)
"""