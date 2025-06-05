import unittest
import numpy as np


def generate_batchs1( discounted_rewards: np.ndarray, batch_size: int):
        have_ones = len(discounted_rewards) % batch_size == 1
        batch_range = np.arange(0, len(discounted_rewards), batch_size, dtype=np.int32)
        indices = np.arange(0, len(discounted_rewards), dtype=np.int32)
        batchs = [indices[i:i+batch_size] for i in batch_range]
        for batch in batchs:
                np.random.shuffle(batch)
        return batchs
def generate_batches( discounted_rewards: np.ndarray, batch_size: int) -> np.ndarray:
        """
        generate many batches of samples from the buffer,
        note that the last batch's length may be smaller than batch_size
        """
        one_left = len(discounted_rewards) % batch_size == 1
        n_states = len(discounted_rewards)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)

        batches = [indices[i:i+batch_size]
                   for i in batch_start if len(indices[i:i+batch_size]) > 1]
        if one_left:
            batches[-1] = np.concatenate((batches[-1], indices[-1:]))
        return batches


# 测试用例
test_cases = [
    # 输入的 discounted_rewards 和 batch_size，以及期望的 batchs 结构
    (np.array([1, 2, 3, 4, 5]), 2, [0, 2, 4], [1, 3]), # batch_size 为偶数
    (np.array([1, 2, 3, 4, 5, 6]), 3, [0, 3], [1, 2, 4, 5]), # batch_size 为奇数但能被长度整除
    (np.array([1, 2, 3, 4, 5, 6, 7]), 3, [0, 3, 6], [1, 2, 4, 5, 7]), # batch_size 为奇数且不能被长度整除
]

for i, (discounted_rewards, batch_size, *expected) in enumerate(test_cases):
        # 生成 batchs
        batchs = generate_batches(discounted_rewards, batch_size)
        # 将生成的 batchs 转换为列表以便比较
        batchs_list = [list(batch) for batch in batchs]
        # 检查生成的 batchs 是否在期望值中
        # self.assertIn(batchs_list, expected)
