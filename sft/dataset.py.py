from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

data_name = "semsum"
data_path = "/root/TinyLlama/sft/emotion/data/"
model_name_or_path = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
IGNORE_TOKEN_ID = -100

#tokenize时作为function调用
def preprocess_data(dataset):
    '''
    {'id': '13818513', 
    'input': "<s>Summarize the dialogue: Amanda: I baked  cookies......, 
    'output': 'Amanda baked cookies and will bring Jerry some tomorrow.</s>'}
    '''
    #Tokenize:分别处理输入和输出
    inputs = [input for input in dataset['input']]
    model_input = tokenizer(
        inputs,
        max_length=512,
        padding='max_length',
        truncation=True,
    ) #[input_ids,attention_mask]
    labels = tokenizer(
        text_target=dataset['output'],
        max_length=128,
        padding='max_length',
        truncation=True,
    )
    '''
    {'id': '13818513', 
    'input_ids': [1, 1, 6991, 3034, 675, 278, 7928, 434, ...], 
    'attention_mask': [1, 1, ....]}
    max input length is 301.
    max output length is 102.
    '''
    # 将[pad]用ignore_token_id代替，计算loss时不考虑
    labels["input_ids"] = [
        [(i if i != tokenizer.pad_token_id else IGNORE_TOKEN_ID) for i in label] 
        for label in labels['input_ids']
    ]
    model_input['labels'] = labels['input_ids']
    return model_input



# 获取tokenizer
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    padding_side="right",
    use_fast=True, # Fast tokenizer giving issues.
    trust_remote_code=True,
)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# dataset = load_dataset(data_name)
# 从本地下载数据集,通过data_files完成数据分片
data_files = {}
for split in ["train","test"]:
    data_files[split] = data_path + split + ".json"
dataset = load_dataset("json", data_files=data_files)

'''
example
{'dialogue': "", 
 'id': "", 
 'summary': ""}
dataset struct:
 DatasetDict({
    train: Dataset({
        features: ['dialogue', 'id', 'summary'],
        num_rows: 14732
    })
    test: Dataset({
        features: ['dialogue', 'id', 'summary'],
        num_rows: 819
    })
})
'''
#TODO:训练集、测试集等划分
#此处在load_dataset时就已经分片

#TODO:数据预处理,可以使用dataset.map函数
#此处仅根据生成式大模型需要,添加[bos][eos]标记和prompt
dataset['train'] = dataset['train'].map(
    lambda x: {
    'input' : tokenizer.bos_token + "Summarize the dialogue: " + x['dialogue'],
    'output': x['summary'] + tokenizer.eos_token,
},remove_columns=['dialogue', 'summary'])
dataset['test'] = dataset['test'].map(
    lambda x: {
    'input' : tokenizer.bos_token + x['dialogue'],
    'output': x['summary'] + tokenizer.eos_token,
},remove_columns=['dialogue', 'summary'])
'''
{'id': '13818513', 
 'input': "<s>Summarize the dialogue: Amanda: I baked  cookies......, 
 'output': 'Amanda baked cookies and will bring Jerry some tomorrow.</s>'}
 '''

tokenized_dataset = dataset.map(preprocess_data, batched=True, remove_columns=['input', 'output'])
#print(tokenized_dataset['train'][0])
#print(tokenized_dataset['test'][0])
tokenized_dataset['train'].save_to_disk("./sft/tokenized_data/train")
tokenized_dataset['test'].save_to_disk("./sft/tokenized_data/test")


'''
# Build the input and labels for causal LM
input_ids = []
labels = []
#input_ids即将token转化为id,用于嵌入向量的查询
#tokenized[input_ids]的输出应该是list(list(id))
#predict_with_generate:输入中不包含target,即学习生成target的能力
#train_on_source:输出包含source,如果不包含则全部用ignore index代替
#demo问题为生成答案的问题,所以为predict_with)generate=true的问题
for source, target in zip(
    tokenized_input['train']['input_ids'] + tokenized_input['test']['input_ids'],
    tokenized_output['train']['input_ids'] + tokenized_output['test']['input_ids']
):
    input_ids.append(torch.tensor(source))

input_ids = pad_sequence(input_ids, batch_first=True, padding_value=-100)
labels = []
data_dict = {
    'input_ids' : input_ids,
    'attention_mask' : input_ids.ne(-100),
}
if labels is not None:
    data_dict['labels'] = labels

'''

