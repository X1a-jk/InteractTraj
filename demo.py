import torch

# 假设vec1和vec2已经被定义，并且vec2包含有效的索引
vec1 = torch.randn(16, 32, 3, 2)  # 随机生成一个(8, 3, 2)的张量
vec2 = torch.randint(0, 3, (16, 32, 1))  # 随机生成一个(8, 1)的索引张量
print(vec1.shape)
print(vec2.dtype)
# 使用torch.gather进行切片操作
# 第二维索引为vec2的值，第三维索引为0（因为我们只取第二维的切片）
sliced_tensor = torch.gather(vec1, 2, vec2.unsqueeze(-1).expand(-1, -1, -1, 2))

# 检查sliced_tensor的尺寸
print(sliced_tensor.shape)  # 应该输出: torch.Size([8, 2])

# 如果你想要得到一个(8, 2)的向量，可以进一步操作
vector = sliced_tensor.squeeze(-2)  # 展平第一维

# 检查vector的尺寸
print(vector.shape)  # 应该输出: torch.Size([8, 2])
