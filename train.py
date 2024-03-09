import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import numpy as np
import matplotlib.pyplot as plt
import time
import torch

# 定义一个用于关系数据集的类，继承自PyTorch的Dataset类
class RelationshipDataset(Dataset):
    def __init__(self, data, node2vec_embeddings, tokenizer, max_length):
        # 初始化函数，接受数据集、Node2Vec嵌入、BERT分词器和最大序列长度作为参数
        self.tokenizer = tokenizer  # BERT分词器
        self.data = data  # 数据集
        self.node2vec_embeddings = node2vec_embeddings  # Node2Vec嵌入
        self.max_length = max_length  # 序列最大长度
        self.embedding_lookup = self.build_embedding_lookup(node2vec_embeddings)  # 建立Node2Vec嵌入的查找表

    def build_embedding_lookup(self, embeddings):
        lookup = {} # 构建Node2Vec嵌入的查找表
        for item in embeddings:
            object1 = item['object1']
            object2 = item['object2']
            # 确保每个object的嵌入只添加一次
            if object1 not in lookup:
                lookup[object1] = item['properties1']
            if object2 not in lookup:
                lookup[object2] = item['properties2']
        return lookup

    def __len__(self):
        return len(self.data)  # 返回数据集的大小

    def __getitem__(self, idx): # 获取数据集中的一个实例，并进行处理
        entry = self.data[idx]
        object1 = entry['object1']
        properties1 = ' '.join(entry['properties1'])
        object2 = entry['object2']
        properties2 = ' '.join(entry['properties2'])
        relationship = entry['relationship']

        # 使用BERT分词器进行编码
        bert_encoding = self.tokenizer.encode_plus(
            f"{object1} {properties1} {object2} {properties2}",
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # 从查找表中获取Node2Vec嵌入
        properties1_vec = torch.tensor(self.embedding_lookup[object1], dtype=torch.float32)
        properties2_vec = torch.tensor(self.embedding_lookup[object2], dtype=torch.float32)

        # 将BERT嵌入和Node2Vec嵌入合并
        input_features = torch.cat([bert_encoding['input_ids'].squeeze(), properties1_vec, properties2_vec])

        return {
            'input_ids': bert_encoding['input_ids'].squeeze(),
            'attention_mask': bert_encoding['attention_mask'].squeeze(),
            'node2vec_embeddings': torch.cat([properties1_vec, properties2_vec], dim=0),  # 将两个嵌入连接起来
            'labels': torch.tensor(relationship, dtype=torch.long)
        }

with open('BERT_Input.json', 'r') as f:
    dataset = json.load(f)  # 从JSON文件中加载数据集

with open('GAE_embedding_Dim768.json', 'r') as f:
    n2v_embedding = json.load(f)  # 从JSON文件中加载Node2Vec嵌入

# 将`relationship`转换为数值格式
unique_relationships = set(tuple(sorted(d['relationship'])) for d in dataset)
relationship_to_idx = {rel: idx for idx, rel in enumerate(unique_relationships)}

# 使用数值标签更新数据集
for d in dataset:
    # 将列表转换为排序后的元组，然后查找
    relationship_tuple = tuple(sorted(d['relationship']))
    d['relationship'] = relationship_to_idx[relationship_tuple]

# 划分数据集
train_data, val_data = train_test_split(dataset, test_size=0.1)  # 划分训练集和验证集

# 初始化BERT分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 128  # 设置最大序列长度

# 创建数据集实例
train_dataset = RelationshipDataset(train_data, n2v_embedding, tokenizer, max_length)
val_dataset = RelationshipDataset(val_data, n2v_embedding, tokenizer, max_length)

# 创建数据加载器
batch_size = 32  # 设置批量大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练数据加载器
val_loader = DataLoader(val_dataset, batch_size=batch_size)  # 验证数据加载器

# 定义模型
class RelationshipClassifier(nn.Module):
    def __init__(self, n_classes, bert_output_size=768, node2vec_size=768, hidden_size=512, num_transformer_layers=2, transformer_heads=8):
        super(RelationshipClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')  # 加载预训练的BERT模型
        # 新增加的降维层，将输入从2304维降到768维
        self.dimension_reduction_layer = nn.Linear(bert_output_size + 2 * node2vec_size, bert_output_size)

        self.fusion_layer = nn.Linear(bert_output_size, hidden_size)  # 注意这里的输入维度改为768
        # Transformer编码器层配置，输入输出都是512
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,  # Transformer的输入和输出维度
            nhead=transformer_heads,  # 多头注意力机制中的头数
            dim_feedforward=hidden_size * 2,  # 前馈网络的维度
            dropout=0.1  # dropout比率
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=num_transformer_layers)  # 定义Transformer编码器
        self.hidden_layer = nn.Linear(hidden_size, hidden_size // 2)  # 定义隐藏层，输入512，输出256
        self.relu = nn.ReLU()  # 定义ReLU激活函数
        self.dropout = nn.Dropout(0.3)  # 定义dropout
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)  # 定义批量归一化层
        self.batchnorm2 = nn.BatchNorm1d(hidden_size // 2)  # 定义批量归一化层
        self.out = nn.Linear(hidden_size // 2, n_classes)  # 定义输出层，输入256，输出分类数

    def forward(self, input_ids, attention_mask, node2vec_embeddings):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # 使用BERT模型处理文本

        pooled_output = bert_output.pooler_output  # 获取BERT模型的池化输出
        # 拼接BERT输出和Node2Vec嵌入，然后通过降维层
        fusion_input = torch.cat([pooled_output, node2vec_embeddings], dim=1)
        reduced_input = self.dimension_reduction_layer(fusion_input)

        x = self.dropout(self.relu(self.fusion_layer(reduced_input)))  # 注意这里使用reduced_input
        x = x.unsqueeze(0)  # Transformer期望的输入是 (S, N, E) 形式，增加一个批次维度
        x = self.transformer_encoder(x)  # 通过Transformer编码器层
        x = x.squeeze(0)  # 移除批次维度
        x = self.batchnorm1(x)  # 应用批量归一化
        x = self.dropout(self.relu(self.hidden_layer(x)))  # 通过隐藏层，ReLU激活函数，和dropout
        x = self.batchnorm2(x)  # 应用批量归一化

        return self.out(x)  # 通过输出层返回分类结果

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否使用CUDA，即是否使用GPU
model = RelationshipClassifier(len(relationship_to_idx)).to(device)  # 实例化模型并根据设备转移
loss_fn = nn.CrossEntropyLoss().to(device)  # 定义损失函数，并根据设备转移
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)  # 定义优化器

# For plotting
train_losses = []  # 用于存储训练损失
train_accuracies = []  # 用于存储训练准确率
val_losses = []  # 用于存储验证损失
val_accuracies = []  # 用于存储验证准确率

def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
    model.train()  # 设置模型为训练模式
    losses = []  # 存储损失
    correct_predictions = 0  # 正确预测的数量
    start_time = time.time()  # 记录开始时间

    for d in data_loader:
        # Adjust to use 'input_features' instead of 'input_ids'
        input_features = d["input_ids"].to(device)  # 获取输入特征，并转移到设备
        attention_mask = d["attention_mask"].to(device)  # 获取注意力掩码，并转移到设备
        node2vec_embeddings = d["node2vec_embeddings"].to(device)  # 获取node2vec嵌入，并转移到设备
        labels = d["labels"].to(device)  # 获取标签，并转移到设备
        # Modify the model call to match the input structure
        outputs = model(input_features, attention_mask, node2vec_embeddings)  # 调用模型进行前向传播
        _, preds = torch.max(outputs, dim=1)  # 获取预测结果
        loss = loss_fn(outputs, labels)  # 计算损失
        correct_predictions += torch.sum(preds == labels)  # 计算正确预测的数量
        losses.append(loss.item())  # 存储损失值
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    end_time = time.time()  # 记录结束时间
    training_time = (end_time - start_time) / 60  # 计算并转换为分钟
    print(f"Training time: {training_time:.2f} minutes")  # 打印训练时间

    return correct_predictions.double() / n_examples, np.mean(losses)  # 返回准确率和平均损失


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()  # 设置模型为评估模式
    losses = []  # 存储损失
    correct_predictions = 0  # 正确预测的数量
    start_time = time.time()  # 记录开始时间

    with torch.no_grad():  # 不计算梯度，减少内存消耗
        for d in data_loader:  # 遍历数据加载器
            input_ids = d["input_ids"].to(device)  # 获取输入ID，并转移到设备
            attention_mask = d["attention_mask"].to(device)  # 获取注意力掩码，并转移到设备
            node2vec_embeddings = d["node2vec_embeddings"].to(device)  # 获取node2vec嵌入，并转移到设备
            labels = d["labels"].to(device)  # 获取标签，并转移到设备
            outputs = model(input_ids, attention_mask, node2vec_embeddings)  # 调用模型进行前向传播
            _, preds = torch.max(outputs, dim=1)  # 获取预测结果
            loss = loss_fn(outputs, labels)  # 计算损失
            correct_predictions += torch.sum(preds == labels)  # 计算正确预测的数量
            losses.append(loss.item())  # 存储损失值
    end_time = time.time()  # 记录结束时间
    eval_time = (end_time - start_time) / 60  # 计算并转换为分钟
    print(f"Evaluation time: {eval_time:.2f} minutes")  # 打印验证时间
    return correct_predictions.double() / n_examples, np.mean(losses)  # 返回准确率和平均损失

# 在开始训练前定义文档路径
train_acc_file = 'train_acc.txt'
train_loss_file = 'train_loss.txt'
val_acc_file = 'val_acc.txt'
val_loss_file = 'val_loss.txt'

# Initialize variables to track the best metrics
best_train_acc = 0
best_val_acc = 0
lowest_train_loss = float('inf')
lowest_val_loss = float('inf')

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        len(train_dataset)
    )

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
        model,
        val_loader,
        loss_fn,
        device,
        len(val_dataset)
    )
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # Update best and lowest metrics
    if train_acc > best_train_acc:
        best_train_acc = train_acc
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    if train_loss < lowest_train_loss:
        lowest_train_loss = train_loss
    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss

    print(f'Validation loss {val_loss} accuracy {val_acc}')
    # 将结果和对应的epoch写入文档
    with open(train_acc_file, 'a') as f:
        f.write(f'Epoch {epoch + 1}: {train_acc}\n')
    with open(train_loss_file, 'a') as f:
        f.write(f'Epoch {epoch + 1}: {train_loss}\n')
    with open(val_acc_file, 'a') as f:
        f.write(f'Epoch {epoch + 1}: {val_acc}\n')
    with open(val_loss_file, 'a') as f:
        f.write(f'Epoch {epoch + 1}: {val_loss}\n')

### 5. Save the Model
torch.save(model.state_dict(), 'relationship_classifier_model_transformer.pth')

# After training loop, print the best metrics
print(f"Best Training Accuracy: {best_train_acc}")
print(f"Best Validation Accuracy: {best_val_acc}")
print(f"Lowest Training Loss: {lowest_train_loss}")
print(f"Lowest Validation Loss: {lowest_val_loss}")

# Save the best metrics to their respective text documents
with open('best_metrics.txt', 'w') as f:
    f.write(f"Best Training Accuracy: {best_train_acc}\n")
    f.write(f"Best Validation Accuracy: {best_val_acc}\n")
    f.write(f"Lowest Training Loss: {lowest_train_loss}\n")
    f.write(f"Lowest Validation Loss: {lowest_val_loss}\n")

# Visualization and Saving Plots
def plot_and_save(data, title, ylabel, save_filename):
    plt.figure(figsize=(10, 6))
    if torch.is_tensor(data[0]):  # 检查列表中的第一个元素是否为张量
        data = [d.cpu().numpy() if d.is_cuda else d.numpy() for d in data]  # 将所有CUDA张量转换为NumPy数组
    plt.plot(data, label=ylabel)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_filename)
    plt.close()

plot_and_save(train_losses, 'Training Loss', 'Loss', 'train_loss.png')
plot_and_save(train_accuracies, 'Training Accuracy', 'Accuracy', 'train_accuracy.png')
plot_and_save(val_losses, 'Validation Loss', 'Loss', 'val_loss.png')
plot_and_save(val_accuracies, 'Validation Accuracy', 'Accuracy', 'val_accuracy.png')
