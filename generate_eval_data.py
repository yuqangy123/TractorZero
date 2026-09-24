import argparse
import pickle
import numpy as np

# 两副牌共 108 张，编号 0~107（与 env/botzone.py 的发牌表示一致）。
# 按级数（2 到 A）生成固定发牌数据，供模型评价时按级数顺序注入。
__DECK_SIZE__ = 108
__PLAYER_COUNT__ = 4
__HAND_CARD_NUM__ = 25
__PUBLIC_CARD_NUM__ = 8
# 级数顺序（10 记为 '0'）
__POINT__ = ['2', '3', '4', '5', '6', '7', '8', '9', '0', 'J', 'Q', 'K', 'A']


def deal_cards():
    """随机发一副牌：4 名玩家各 25 张 + 8 张底牌。"""
    allo = list(range(__DECK_SIZE__))
    np.random.shuffle(allo)
    allocation = [
        allo[i * __HAND_CARD_NUM__:(i + 1) * __HAND_CARD_NUM__]
        for i in range(__PLAYER_COUNT__)
    ]
    publiccard = allo[
        __PLAYER_COUNT__ * __HAND_CARD_NUM__:
        __PLAYER_COUNT__ * __HAND_CARD_NUM__ + __PUBLIC_CARD_NUM__
    ]
    return {'allocation': allocation, 'publiccard': publiccard}


def get_parser():
    parser = argparse.ArgumentParser(description='Tractor: eval deal data generator')
    parser.add_argument('--output', default='eval_data', type=str)
    parser.add_argument('--num_deals', default=100, type=int,
                        help='每个级数生成几局牌（默认 100）')
    return parser


def generate(num_deals):
    data = {}
    for level in __POINT__:
        data[level] = [deal_cards() for _ in range(num_deals)]
    return data


if __name__ == '__main__':
    flags = get_parser().parse_args()
    output_pickle = flags.output + '.pkl'
    print(f"output_pickle: {output_pickle}")
    print(f"generating {flags.num_deals} deals per level for levels: {__POINT__}")

    data = generate(flags.num_deals)

    with open(output_pickle, 'wb') as g:
        pickle.dump(data, g, pickle.HIGHEST_PROTOCOL)

    print(f"saved {len(data)} levels x {flags.num_deals} deals -> {output_pickle}")