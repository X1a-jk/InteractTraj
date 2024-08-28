import torch

# 假设vec1和vec2已经被定义
vec1 = torch.randn(16, 32, 3, 8)  # 随机生成一个(16, 32, 3, 8)的张量
vec2 = torch.randint(0, 3, (16, 32, 1))  # 随机生成一个(16, 32, 1)的索引张量

# 扩展vec2的形状以匹配vec1的第二维
vec2_expanded = vec2.expand(-1, -1, 8)

# 使用torch.gather进行切片操作
# 注意：我们需要在第二维上进行切片，所以dim=2
# 由于vec2_expanded已经扩展到(16, 32, 3)，我们可以直接使用它来选择vec1的第二维
sliced_tensor = torch.gather(vec1, 2, vec2_expanded)

# 检查sliced_tensor的尺寸
print(sliced_tensor.shape)  # 应该输出: torch.Size([16, 32, 8])
