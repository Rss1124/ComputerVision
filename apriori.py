from functools import reduce
from typing import Dict, List
from copy import deepcopy

def Apriori(transactions : dict, min_sup : float, level : int = -1) -> Dict[str, List[List[str]]]:
    cur_collection = {}
    n_sup = int(len(transactions) * min_sup)

    # get all the unique element in the itemset
    unique_items = reduce(lambda x, y : x | y, (set(items) for items in transactions.values()))
    # initialise the C
    C = [[item] for item in unique_items]
    C.sort(key=lambda x : x[0])

    itemset_list = [set(itemset) for itemset in transactions.values()]

    for iter_num, _ in enumerate(unique_items):
        # step1: count and filter the frequent itemsets
        K = list()
        for i, item in enumerate(C):
            item = set(item)
            count = 0
            for itemset in itemset_list:
                if item & itemset == item:   # this way, item is subset to itemset
                    count += 1
            if count >= n_sup:
                item = list(item)
                item.sort()         # sort is necessary
                K.append(item)
        
        if len(K):
            cur_collection["L" + str(iter_num + 1)] = K
            if level > 0 and iter_num + 1 == level:
                return cur_collection
        
        # step2: check and connect between frequent itemsets
        K_l = len(K)
        no_trim_C = set()
        for i in range(K_l - 1):
            for j in range(i + 1, K_l):
                if K[i][:-1] == K[j][:-1]:
                    temp = deepcopy(K[i][:-1])
                    a, b = K[i][-1], K[j][-1]
                    temp += [a, b] if a < b else [b, a]
                    no_trim_C.add(tuple(temp))   # list is not hashable
        if len(no_trim_C) == 0:
            break

        # step3: truncate the L
        if iter_num == 0:
            C = [list(itemset) for itemset in no_trim_C]
            continue

        C = []
        K = {tuple(itemset) for itemset in K}
        for itemset in no_trim_C:
            itemset = sorted(list(itemset))
            for item in itemset:
                temp_itemset = deepcopy(itemset)
                temp_itemset.remove(item)
                temp_itemset = tuple(temp_itemset)
                if temp_itemset not in K:
                    break
            else:
                C.append(itemset)

        # new iteration
        if len(C) == 0:
            break

    return cur_collection

if __name__ == "__main__":
    transactions = {
        "T1" : ["鸡蛋", "牙膏", "牛排", "牛奶", "面包"],
        "T2" : ["鸡蛋", "亚麻籽", "橄榄油", "牛奶", "面包"],
        "T3" : ["鸡蛋", "泡芙", "奶油", "牛奶", "面包"],
        "T4" : ["鸡蛋", "低筋面粉", "糖粉", "黄油", "牛奶"],
        "T5" : ["牙膏", "牙刷", "毛巾", "洗面奶"],
        "T6" : ["牛排", "黄油", "黑椒酱"]
    }

    result = Apriori(transactions, 0.5, -1)
    for k in result:
        print(k, ":")
        for v in result[k]:
            print("({})".format(' '.join(v)), end="  ")
        print()