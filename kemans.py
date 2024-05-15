import numpy as np
import matplotlib.pyplot as plt

class item:
    def __init__(self, index, cood, center_idx, distance_to_center):
        self.index = index
        self.cood = cood
        self.center_idx = center_idx
        self.distance_to_center = distance_to_center

def calc_distance(p_a, p_b):
    return (p_a[0] - p_b[0])**2 + (p_a[1] - p_b[1])**2


def kmeans(data, kmeans_type=3, epoch=1000):
    total_dis = 0
    center=[]
    for i in range(kmeans_type):
        center.append(data[i])
    temp_cood = np.zeros([kmeans_type, 2])
    temp_cnt = np.zeros(kmeans_type)

    l = len(data)
    data_set = []
    for i in range(l):
        it = item(i, data[i], -1, -1)
        data_set.append(it)

    for k in range(epoch):
        for it in data_set:
            for i,c in enumerate(center):
                dis = calc_distance(it.cood, c)           
                if it.distance_to_center == -1 or dis < it.distance_to_center:
                    it.distance_to_center = dis
                    it.center_idx = i
            if k==epoch-1:
                total_dis += it.distance_to_center
            temp_cood[it.center_idx] += it.cood
            temp_cnt[it.center_idx] += 1

        for i in range(kmeans_type):
            center[i] = temp_cood[i] / temp_cnt[i]
            #print("new center {} = {}".format(i, center[i]))
    x, y, color= [], [], []
    for it in data_set:
        x.append(it.cood[0])
        y.append(it.cood[1])
        color.append(it.center_idx*10)
    plt.scatter(x, y, c=color, cmap='viridis')
    plt.show()
    return total_dis

if __name__ == "__main__":
    data = np.random.randn(100,2)
    k_list = [1,2,3,4,5,6,7,8]
    different_result_of_k = []
    for k in k_list:
        different_result_of_k.append(kmeans(data, kmeans_type=k))
    x = range(1, len(different_result_of_k) + 1)
    plt.plot(x, different_result_of_k, marker='o')
    plt.title('Line Graph with 10 Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()