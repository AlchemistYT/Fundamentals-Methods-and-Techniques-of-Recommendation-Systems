import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from collections import defaultdict


class MultiRelationalAttentionConv(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super(MultiRelationalAttentionConv, self).__init__()
        self.embed_dim = embed_dim
        # 论文公式6/7/8：可训练权重矩阵
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_uv = nn.Linear(embed_dim, embed_dim)
        self.W_u = nn.Linear(embed_dim, embed_dim)
        self.W_uu = nn.Linear(embed_dim, embed_dim)
        self.W_vv = nn.Linear(embed_dim, embed_dim)

        # 论文公式9：关系权重矩阵
        self.W1 = nn.Linear(embed_dim, embed_dim)  # 协作关系
        self.W2 = nn.Linear(embed_dim, embed_dim)  # 交互关系
        self.W3 = nn.Linear(embed_dim, embed_dim)  # 聚合权重

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def calc_attention(self, embed_src, embed_dst, rel_score, src_deg, dst_deg):
        """论文公式10/11/12：多关系注意力分数计算"""
        # 消息构建
        msg = embed_dst + embed_src * embed_dst  # 元素乘 ⊙
        # 注意力分数：关系强度 + 嵌入相似度 + 度归一化
        att_score = (rel_score / torch.sqrt(src_deg * dst_deg + 1e-8)) * \
                    (torch.matmul(embed_src.unsqueeze(1), msg.unsqueeze(-1))).squeeze() / \
                    torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32))
        # Softmax归一化
        att_score = F.softmax(att_score, dim=-1)
        return att_score, msg

    def forward(self, embed, adj_collab, adj_interact, adj_similar,
                collab_score, interact_score, similar_score,
                deg_dev, deg_task):
        """
        嵌入传播前向计算
        embed: [N, embed_dim] → 拼接的开发者+任务嵌入
        adj_collab: 开发者-开发者协作邻接
        adj_interact: 开发者-任务交互邻接
        adj_similar: 任务-任务相似邻接
        """
        num_dev = adj_collab.shape[0]
        # 1. 开发者侧：协作关系 + 交互关系传播
        dev_embed = embed[:num_dev]
        # 协作关系消息传播
        collab_att, collab_msg = self.calc_attention(dev_embed, dev_embed, collab_score, deg_dev, deg_dev)
        collab_prop = torch.matmul(collab_att, self.W1(collab_msg))

        # 交互关系消息传播
        task_embed = embed[num_dev:]
        interact_att, interact_msg = self.calc_attention(dev_embed, task_embed, interact_score, deg_dev, deg_task)
        interact_prop = torch.matmul(interact_att, self.W2(interact_msg))

        # 开发者嵌入聚合
        dev_prop = self.leaky_relu(self.W3(dev_embed + collab_prop + interact_prop))

        # 2. 任务侧：相似关系传播
        similar_att, similar_msg = self.calc_attention(task_embed, task_embed, similar_score, deg_task, deg_task)
        similar_prop = torch.matmul(similar_att, self.W3(task_embed))
        task_prop = self.leaky_relu(self.W3(task_embed + similar_prop))

        # 拼接传播后的嵌入
        new_embed = torch.cat([dev_prop, task_prop], dim=0)
        return self.dropout(new_embed)


class DevRec(nn.Module):
    def __init__(self, num_dev, num_task, embed_dim=64, num_layers=3, dropout=0.1):
        super(DevRec, self).__init__()
        self.num_dev = num_dev
        self.num_task = num_task
        self.embed_dim = embed_dim
        self.num_layers = num_layers  # 论文最优L=3

        # 嵌入层：论文5.1节
        self.dev_embed = nn.Embedding(num_dev, embed_dim)
        self.task_embed = nn.Embedding(num_task, embed_dim)

        # 多关系注意力卷积层（高阶传播）
        self.conv_layers = nn.ModuleList([
            MultiRelationalAttentionConv(embed_dim, dropout) for _ in range(num_layers)
        ])

        # 初始化嵌入
        self._init_weights()

    def _init_weights(self):
        """Xavier初始化（对齐论文6.1.4）"""
        nn.init.xavier_uniform_(self.dev_embed.weight)
        nn.init.xavier_uniform_(self.task_embed.weight)

    def forward(self, adj_collab, adj_interact, adj_similar,
                collab_score, interact_score, similar_score,
                deg_dev, deg_task):
        """前向传播：高阶嵌入传播 + 层拼接"""
        # 初始嵌入
        dev_embed = self.dev_embed.weight
        task_embed = self.task_embed.weight
        embed = torch.cat([dev_embed, task_embed], dim=0)
        all_embeds = [embed]  # 存储各层嵌入（论文公式18）

        # 高阶嵌入传播（论文5.2.2节）
        for conv in self.conv_layers:
            embed = conv(embed, adj_collab, adj_interact, adj_similar,
                         collab_score, interact_score, similar_score, deg_dev, deg_task)
            all_embeds.append(embed)

        # 拼接所有层嵌入（论文公式18）
        final_embed = torch.cat(all_embeds, dim=-1)
        final_dev_embed = final_embed[:self.num_dev]
        final_task_embed = final_embed[self.num_dev:]

        return final_dev_embed, final_task_embed

    def predict(self, dev_idx, task_idx, dev_embed, task_embed):
        """预测：内积计算匹配度（论文公式19）"""
        dev_e = dev_embed[dev_idx]
        task_e = task_embed[task_idx]
        return torch.sum(dev_e * task_e, dim=-1)


class DevRecLoss(nn.Module):
    def __init__(self, lambda_reg=1e-5):
        super(DevRecLoss, self).__init__()
        self.lambda_reg = lambda_reg

    def bpr_loss(self, pos_score, neg_score):
        """BPR损失（论文公式20）"""
        return -torch.mean(F.logsigmoid(pos_score - neg_score))

    def forward(self, pos_pred, neg_pred, collab_pos, collab_neg,
                similar_pos, similar_neg, model_params):
        """
        联合损失：L_DevRec = L_col + L_sim + L_pre + L2正则
        L_pre: 预测损失 | L_col: 协作损失 | L_sim: 相似损失
        """
        # 预测损失
        l_pre = self.bpr_loss(pos_pred, neg_pred)
        # 协作关系损失
        l_col = self.bpr_loss(collab_pos, collab_neg)
        # 任务相似损失
        l_sim = self.bpr_loss(similar_pos, similar_neg)
        # L2正则
        l_reg = 0
        for param in model_params:
            l_reg += torch.sum(torch.pow(param, 2))
        l_reg = self.lambda_reg * l_reg

        return l_pre + l_col + l_sim + l_reg


def main():
    # 超参数（对齐论文6.1.4）
    NUM_DEV = 100  # 开发者数量
    NUM_TASK = 200  # 任务数量
    EMBED_DIM = 64  # 嵌入维度
    NUM_LAYERS = 3  # 高阶传播层数
    BATCH_SIZE = 512
    LR = 1e-3
    EPOCHS = 50
    LAMBDA_REG = 1e-5

    # 1. 模拟多关系图数据（论文4节定义）
    # 邻接矩阵
    adj_collab = torch.randn(NUM_DEV, NUM_DEV)  # 开发者协作
    adj_interact = torch.randn(NUM_DEV, NUM_TASK)  # 开发者-任务交互
    adj_similar = torch.randn(NUM_TASK, NUM_TASK)  # 任务相似
    # 关系分数
    collab_score = torch.rand(NUM_DEV, NUM_DEV)
    interact_score = torch.rand(NUM_DEV, NUM_TASK)
    similar_score = torch.rand(NUM_TASK, NUM_TASK)
    # 节点度
    deg_dev = torch.sum(adj_collab, dim=-1) + torch.sum(adj_interact, dim=-1)
    deg_task = torch.sum(adj_similar, dim=-1) + torch.sum(adj_interact.T, dim=-1)

    # 2. 模型初始化
    model = DevRec(NUM_DEV, NUM_TASK, EMBED_DIM, NUM_LAYERS)
    loss_fn = DevRecLoss(LAMBDA_REG)
    optimizer = Adam(model.parameters(), lr=LR)

    # 3. 训练循环
    model.train()
    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # 前向传播
        dev_embed, task_embed = model(adj_collab, adj_interact, adj_similar,
                                      collab_score, interact_score, similar_score,
                                      deg_dev, deg_task)

        # 模拟正负样本
        dev_idx = torch.randint(0, NUM_DEV, (BATCH_SIZE,))
        pos_task = torch.randint(0, NUM_TASK, (BATCH_SIZE,))
        neg_task = torch.randint(0, NUM_TASK, (BATCH_SIZE,))

        # 预测分数
        pos_pred = model.predict(dev_idx, pos_task, dev_embed, task_embed)
        neg_pred = model.predict(dev_idx, neg_task, dev_embed, task_embed)

        # 模拟协作/相似正负样本
        collab_pos = torch.rand(BATCH_SIZE)
        collab_neg = torch.rand(BATCH_SIZE)
        similar_pos = torch.rand(BATCH_SIZE)
        similar_neg = torch.rand(BATCH_SIZE)

        # 计算损失
        loss = loss_fn(pos_pred, neg_pred, collab_pos, collab_neg,
                       similar_pos, similar_neg, model.parameters())

        # 反向传播
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}")

    # 4. 推理：为指定开发者推荐Top-K任务
    model.eval()
    with torch.no_grad():
        dev_embed, task_embed = model(adj_collab, adj_interact, adj_similar,
                                      collab_score, interact_score, similar_score,
                                      deg_dev, deg_task)
        target_dev = 0
        scores = torch.matmul(dev_embed[target_dev].unsqueeze(0), task_embed.T).squeeze()
        topk_tasks = torch.topk(scores, k=5).indices
        print(f"为开发者{target_dev}推荐的Top-5任务: {topk_tasks.numpy()}")


if __name__ == "__main__":
    main()