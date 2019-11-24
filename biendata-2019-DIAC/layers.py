from torch.utils.data import Dataset

# 训练集 验证集自定义Dataset
class Train_Dataset(Dataset):
    def __init__(self, bert_list):
        self.dataset = bert_list

    def __getitem__(self, item):
        text = self.dataset[item][0]
        label = int(self.dataset[item][1])
        
        return text, label

    def __len__(self):
        return len(self.dataset)

# 测试集自定义Dataset
class Test_Dataset(Dataset):
    def __init__(self, bert_list):
        self.dataset = bert_list
    
    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)