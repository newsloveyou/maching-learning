'''
Update on 2023-05-30
Author yyw
'''

import math
from collections import Counter
def createDataSet():

        # 数据
        dataSet = [
            [0, 0, 0, 0, 'no'],
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no'],
        ]
        # 列名
        labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
        return dataSet, labels


#Shannon entropy
def calcShannonEnt(dataSet):#计算给定分类的香农熵,计算出来的是信息熵
    #     Args:
    #     dataSet 数据集
    # Returns:
    #     返回 每一组feature下的某个分类下，香农熵的信息期望


    # -----------计算香农熵的第一种实现方式start--------------------------------------------------------------------------------
    # 求list的长度，表示计算参与训练的数据量
    numEntries = len(dataSet)

    # 计算分类标签label出现的次数
    labelCounts = {}
    # the the number of unique elements and their occurance
    for featVec in dataSet:
        # 将当前实例的标签存储，即每一行数据的最后一个数据代表的是标签
        currentLabel = featVec[-1]
        #就是创建映射关系
        # 为所有可能的分类创建字典，如果当前的键值不存在，则扩展字典并将当前键值加入字典。每个键值都记录了当前类别出现的次数。
        #如果此时的类的标签不存在 就创建一个新的桶 负责 该标签所在的桶++
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # 对于label标签的占比，求出label标签的香农熵
    shannonEnt = 0.0
    for key in labelCounts:
        # 使用所有类标签的发生频率计算类别出现的概率。
        prob = float(labelCounts[key]) / numEntries
        # log base 2
        # 计算香农熵，以 2 为底求对数
        shannonEnt -= prob * math.log(prob, 2)
    # -----------计算香农熵的第一种实现方式end--------------------------------------------------------------------------------

    # # -----------计算香农熵的第二种实现方式start--------------------------------------------------------------------------------
    # # 统计标签出现的次数
    # label_count = Counter(data[-1] for data in dataSet)
    # # 计算概率
    # probs = [p[1] / len(dataSet) for p in label_count.items()]
    # # 计算香农熵
    # shannonEnt = sum([-p * log(p, 2) for p in probs])
    # # -----------计算香农熵的第二种实现方式end--------------------------------------------------------------------------------
    return shannonEnt

def splitDataSet(dataSet, index, value):
    """splitDataSet(通过遍历dataSet数据集，求出index对应的colnum列的值为value的行)
        就是依据index列进行分类，如果index列的数据等于 value的时候，就要将 index 划分到我们创建的新的数据集中
    Args:
        dataSet 数据集                 待划分的数据集
        index 表示每一行的index列        划分数据集的特征
        value 表示index列对应的value值   需要返回的特征的值。
    Returns:
        index列为value的数据集【该数据集需要排除index列】
    """
    # -----------切分数据集的第一种方式 start------------------------------------
    retDataSet = []
    for featVec in dataSet:
        # index列为value的数据集【该数据集需要排除index列】
        # 判断index列的值是否为value
        if featVec[index] == value:
            # chop out index used for splitting
            # [:index]表示前index行，即若 index 为2，就是取 featVec 的前 2 行
            reducedFeatVec = featVec[:index]
            '''
            请百度查询一下:  extend和append的区别
            list.append(object) 向列表中添加一个对象object
            list.extend(sequence) 把一个序列seq的内容添加到列表中
            1、使用append的时候，是将new_media看作一个对象，整体打包添加到music_media对象中。
            2、使用extend的时候，是将new_media看作一个序列，将这个序列和music_media序列合并，并放在其后面。
            result = []
            result.extend([1,2,3])
            print result
            result.append([4,5,6])
            print result
            result.extend([7,8,9])
            print result
            结果: 
            [1, 2, 3]
            [1, 2, 3, [4, 5, 6]]
            [1, 2, 3, [4, 5, 6], 7, 8, 9]
            '''
            reducedFeatVec.extend(featVec[index+1:])
            #因为后面构造树的过程中删除了对应的属性 因此这里需要忽略对应的index那一列
            # [index+1:]表示从跳过 index 的 index+1行，取接下来的数据
            # 收集结果值 index列为value的行【该行需要排除index列】
            retDataSet.append(reducedFeatVec)
    # -----------切分数据集的第一种方式 end------------------------------------

    # # -----------切分数据集的第二种方式 start------------------------------------
    # retDataSet = [data for data in dataSet for i, v in enumerate(data) if i == axis and v == value]
    # # -----------切分数据集的第二种方式 end------------------------------------
    return retDataSet


def chooseBestFeatureToSplit(dataSet):
    """chooseBestFeatureToSplit(选择最好的特征)

    Args:
        dataSet 数据集
    Returns:
        bestFeature 最优的特征列
    """

    # -----------选择最优特征的第一种方式 start------------------------------------
    # 求第一行有多少列的 Feature, 最后一列是label列所以减一
    # numFeatures = len(dataSet[0]) - 1
    # # label的信息熵 即最初的信息熵
    # baseEntropy = calcShannonEnt(dataSet)
    # # 最优的信息增益值, 和最优的Featurn编号
    # bestInfoGain, bestFeature = 0.0, -1
    # # 遍历所有特征
    # for i in range(numFeatures):
    #     # create a list of all the examples of this feature
    #     # 获取每一个实例的第i+1个feature，组成list集合
    #     featList = [example[i] for example in dataSet]
    #     # get a set of unique values
    #     # 获取剔重后的集合，使用set对list数据进行去重
    #     uniqueVals = set(featList)
    #     # 创建一个临时的信息熵
    #     newEntropy = 0.0
    #     # 遍历某一列的value集合，计算该列的信息熵
    #     # 遍历当前特征中的所有唯一属性值，对每个唯一属性值划分一次数据集，计算数据集的新熵值，并对所有唯一特征值得到的熵求和。
    #     for value in uniqueVals:
    #         subDataSet = splitDataSet(dataSet, i, value)
    #         prob = len(subDataSet) / float(len(dataSet))
    #         newEntropy += prob * calcShannonEnt(subDataSet)
    #     # gain[信息增益]: 划分数据集前后的信息变化， 获取信息熵最大的值
    #     # 信息增益是熵的减少或者是数据无序度的减少。最后，比较所有特征中的信息增益，返回最好特征划分的索引值。
    #     infoGain = baseEntropy - newEntropy
    #     #print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
    #     if (infoGain > bestInfoGain):
    #         bestInfoGain = infoGain
    #         bestFeature = i
    # return bestFeature
    # -----------选择最优特征的第一种方式 end------------------------------------

    # # -----------选择最优特征的第二种方式 start------------------------------------
    # 计算初始香农熵
    base_entropy = calcShannonEnt(dataSet)
    best_info_gain = 0
    best_feature = -1
    # 遍历每一个特征0
    #python中遍历矩阵的每一列
    #a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # print([i[2] for i in a])

    for i in range(len(dataSet[0]) - 1):
        # 对当前特征进行统计
        #Counter函数  主要功能：可以支持方便、快速的计数，将元素数量统计，然后计数并返回一个字典，键为元素，值为元素个数
        feature_count = Counter([data[i] for data in dataSet])
        #feature[0]是特征    feature[1]是该特征的数目
        # 计算分割后的香农熵
        new_entropy = sum(feature[1] / float(len(dataSet)) * calcShannonEnt(splitDataSet(dataSet, i, feature[0])) \
                       for feature in feature_count.items())
        # 更新值
        info_gain = base_entropy - new_entropy
        #print('No. {0} feature info gain is {1:.3f}'.format(i, info_gain))
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature
    # # -----------选择最优特征的第二种方式 end------------------------------------

def majorityCnt(curLabelList):
    # 获取当前样本里最多的标签
        classCount = {}
        maxKey, maxValue = None, None
        for label in curLabelList:
            if label in classCount.keys():
                classCount[label] += 1
                if maxValue < classCount[label]:
                    maxKey, maxValue = label, classCount[label]
            else:
                classCount[label] = 1
                if maxKey is None:
                    maxKey, maxValue = label, 1

        return maxKey


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
     #判断当前节点的样本的标签是不是已经全都是1个值了 如果是就返回这个唯一的类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
   #判断当前可划分的特征数是否为1 如果是1就返回当前样本里面最多的那个类别
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 1.选择最好的特征进行划分 返回值是索引
    bestFeat = chooseBestFeatureToSplit(dataSet)
    # 2.获取label的名称
    bestFeatLabel = labels[bestFeat]
    #  3.构造当前节点
    myTree = {bestFeatLabel: {}}
    #  4.删除当前被选中的特征
    del (labels[bestFeat])
    # 5.获取当前最佳的特征的那一列
    featValues = [example[bestFeat] for example in dataSet]
    #6.去重
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余的标签label
        subLabels = labels[:]
        # 遍历当前选择特征包含的所有属性值，在每个数据集划分上递归调用函数createTree()
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


def classify():
   dataSet,labels=createDataSet()
   myDecisionTree=createTree(dataSet,labels)
   print(myDecisionTree)
   print(get_tree_height(myDecisionTree))

def get_tree_height(tree):

    if not isinstance(tree, dict):
        return 1

    child_trees = list(tree.values())[0].values()

    # 遍历子树, 获得子树的最大高度
    max_height = 0
    for child_tree in child_trees:
        child_tree_height = get_tree_height(child_tree)

        if child_tree_height > max_height:
            max_height = child_tree_height

    return max_height + 1


if __name__ == "__main__":
    classify()