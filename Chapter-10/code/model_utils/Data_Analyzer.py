

# map item_id to item_str @

# 统计 item-item 共现频率 @

# store item_2_instance_ids @

# handcraft some unreliable instances
    # 人为的 manipulate 一些 instance
        # 人为构造 false negative: 删除某些原本频繁共现的 item 的 instance，模拟 data sparsity 导致的 false negative
        # 人为构造 false positive: 添加一些 instance，其中令某些不相关 item 之间经常共现，模拟某两个 item 一起打折或上排行榜的 产生的 false positive

    # 记录被修改 instance 的 index
    # 在原数据集上训练一个模型，看它对这些 instance 的分类结果
    # 在修改后的数据集上训练另一个模型，看它对这些 instance 的分类结果。

    # 人为构造 false negative: 删除某些原本频繁共现的 item 的 instance，模拟 data sparsity 导致的 false negative， 即人为拉低两个 高共现 item pair 之间的共现频率 （人为构造低共现 item pair）

    # 人为构造 false positive: 添加一些 instance，其中令某些不相关 item 之间经常共现，即人为构造高共现 item-pair，模拟某两个 item 一起打折或上排行榜的 产生的 false positive，人为拉高两个 低共现 item pair 之间的共现频率





#
# 观察 new model 对修改后 instance 的分类，和 old model 对未修改 instance 的分类，若前者 unreliable instance 比例更高，说明包含 unfrequently co-ocurred item pairs 的 instance 更容易被识别为 unreliable，即

    # 观察 new model 对修改后 instance 的分类， 和 old model 对未修改 instance 的分类，若前者 unreliable instance 比例更高，说明包含 unfrequently co-ocurred item pairs 的 instance 更容易被识别未 unreliable


# get instances that are classified as unreliable









# new model 训练集：修改某些原本高共现的 item-pair 对应的 instance，另这些 item-pair 不再频繁共现（人为构造低共现 item-pair）；添加大量 instance，令某些低共现 item pair 变为高共现（人为构造高共现 item-pair）

# old model 训练集：添加少量 instance，这些 instance 中包含 new model 训练集中的原低共现 item pair

# 人为构造 高/低共现 item pair 是针对 new model 来讲

# 分别在两个模型各自训练集中，获取人为构造低共现 item pair 的 instance idx，观察 new model 和 old model 对其的分类 (12% and 2%, respectively)，前者 ureliable 比例高，说明包含 低共现 item-pair 的 instance 更容易被识别为 unreliable（false negative）

# 分别在两个模型各自训练集中，获取人为构造高共现 item pair 的 instance idx，观察 new model 和 old model 对其的分类 (0% and 0%, maybe the code has bugs)，若前者 ureliable 比例低，说明包含 高共现 item-pair 的 instance 更容易被识别为 reliable (false positive)


# To do list



