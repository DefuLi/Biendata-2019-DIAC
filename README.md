# 基于Adversarial Attack的问题等价性判别比赛

## 赛题简介
虽然近年来智能对话系统取得了长足的进展，但是针对专业性较强的问答系统（如法律、政务等），如何准确的判别用户的输入是否为给定问题的语义等价问法仍然是智能问答系统的关键。举例而言，“市政府管辖哪些部门？”和“哪些部门受到市政府的管辖？”可以认为是语义上等价的问题，而“市政府管辖哪些部门？”和“市长管辖哪些部门？”则为不等价的问题。<br>

该[比赛](https://biendata.com/competition/2019diac/)的主要任务是判断两个语句是否为语义等价，这两个语句可能从表现形式上极其相似。

## 基于bert的分类
本程序采用bert作为预训练模型，使用PyTorch深度学习框架，调用的第三方库为huggingface的transformers库，该库可以很方便的对bert进行调用并进行分类。<br>

调用transformers库的bert模型示例：
```
from transformers import BertTokenizer, BertConfig, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)
loss, logits = outputs[:2]
```

以上是transformers库给出的调用中文版bert对句子进行分类的示例，在我们这个比赛中，输入的文本是一对语句。所以输入应该拼接成如下形式：
```
text = [CLS]市政府管辖哪些部门？[SEP]哪些部门受到市政府的管辖？[SEP]
```

## 结果
本程序对数据集进行了5折划分，每一折进行3个epoch训练。选出每一折训练中的最优model进行保存测试。

最终结果是5个模型的投票，少数服从多数原则。

测试结果f1值在0.89左右。
