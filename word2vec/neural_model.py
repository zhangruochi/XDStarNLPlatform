import jieba
jieba.load_userdict("userdict.txt")  # 加载自定义词典

import math
from string import punctuation
from heapq import nlargest
from itertools import product, count
from gensim.models import word2vec
import numpy as np



def get_stop_words_set(stop_words_path):
    """读取停用词表
    """
    with open(stop_words_path, 'r', encoding="utf8") as file:
        return set([line.strip() for line in file])


def segmentation(docs_path, stop_words_set):
    """分词
    """
    punctuations = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+' '"
    corpus = []
    index = 0

    segmentation_file = open("information.seg.txt", mode="w", encoding="utf8")

    with open(docs_path, mode="r", encoding="utf8") as docs:

        for line in docs.readlines():
            seg_list = jieba.cut(line, cut_all=False)
            tmp_list = []
            for term in seg_list:
                if term not in punctuations and term not in stop_words_set:
                    tmp_list.append(term)
            if len(tmp_list) == 0:
                continue
            segmentation_file.write(" ".join(tmp_list) + "\n")
            index += 1
            if index % 1000 == 0:
                print(index)

    segmentation_file.close()


def train_model():
    """训练词向量神经网络模型
    """
    stop_words_path = "stopwords.txt"
    docs_path = "information.txt"
    segmentation_docs_path = "information.seg.txt"

    stop_words_set = get_stop_words_set(stop_words_path)
    segmentation(docs_path, stop_words_set)

    # 模型训练，生成词向量
    sentences = word2vec.LineSentence(segmentation_docs_path)
    model = word2vec.Word2Vec(
        sentences, size=400, window=5, min_count=5, workers=4)
    model.save('xdstar_information.lexicon')



def cut_sentences(sentence):
    """分句
    """
    puns = frozenset(u'。！？；')
    tmp = []
    for ch in sentence:
        tmp.append(ch)
        if puns.__contains__(ch):
            yield ''.join(tmp)
            tmp = []
    yield ''.join(tmp)

"""
def two_sentences_similarity(sents_1, sents_2):
    """计算两个句子的相似度
    """
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    return counter / (math.log(len(sents_1) + len(sents_2)))
"""

def create_graph(word_sent):
    """传入句子链表  返回句子之间相似度的图
    """
    num = len(word_sent)
    board = [[0.0 for _ in range(num)] for _ in range(num)]

    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = compute_similarity_by_avg(word_sent[i], word_sent[j])
    return board


def cosine_similarity(vec1, vec2):
    """向量间余弦相似度的定义
    """
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


def compute_similarity_by_avg(sents_1, sents_2):
    """计算两个句子的相似度
    """
    if len(sents_1) == 0 or len(sents_2) == 0:
        return 0.0
    vec1 = model[sents_1[0]]
    for word1 in sents_1[1:]:
        vec1 = vec1 + model[word1]

    vec2 = model[sents_2[0]]
    for word2 in sents_2[1:]:
        vec2 = vec2 + model[word2]

    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
    return similarity


def calculate_score(weight_graph, scores, i):
    """计算句子的分数
    """
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
            if denominator == 0:
                denominator = 1
        added_score += fraction / denominator
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score


def weight_sentences_rank(weight_graph):
    """输入相似度的图（矩阵),返回各个句子的分数
    """
    # 初始分数设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]

    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores


def different(scores, old_scores):
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            flag = True
            break
    return flag


def filter_symbols(sents):
    """去停用词和标点符号
    """
    stopwords = list(get_stop_words_set("stopwords.txt")) + \
        list("\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+' '〔〕")
    _sents = []
    for sentence in sents:
        tmp_sentence = []
        for word in sentence:
            if word not in stopwords:
                tmp_sentence.append(word)
        if tmp_sentence:
            _sents.append(tmp_sentence)
    return _sents



def filter_model(sents):
    """
    如果词不在模型里，则不做计算
    """
    _sents = []
    for sentence in sents:
        tmp_sentence = []
        for word in sentence:
            if word in model:
                tmp_sentence.append(word)
        if tmp_sentence:
            _sents.append(tmp_sentence)
    return _sents

def calcute_sentences_distance(sentence1, sentence2):
    """计算句子间的相似性
    """

    tmp1 = list(jieba.cut(sentence1, cut_all=False))
    sentence1 = [word for word in tmp1 if word in model]

    tmp2 = list(jieba.cut(sentence2, cut_all=False))
    sentence2 = [word for word in tmp2 if word in model]

    similarity = compute_similarity_by_avg(sentence1, sentence2)

    return similarity


def summarize(text, n):
    tokens = cut_sentences(text)
    sentences = []
    sents = []
    for sent in tokens:
        sentences.append(sent)
        sents.append([word for word in jieba.cut(sent) if word])

    sents = filter_symbols(sents)
    sents = filter_model(sents)
    graph = create_graph(sents)

    scores = weight_sentences_rank(graph)
    sent_selected = nlargest(n, zip(scores, count()))
    sent_index = []
    for i in range(n):
        sent_index.append(sent_selected[i][1])
    return [sentences[i] for i in sent_index]


def test_model():
    """
    print("")
    print("most_similar of 安全")
    print(model.most_similar("安全"))
    """

    print("")
    print("微积分 vs 数学: " + str(model.similarity("微积分", "数学")))
    print("微积分 vs 衣服: " + str(model.similarity("微积分", "衣服")))
    print("孙悟空 vs 猪八戒: " + str(model.similarity("孙悟空", "猪八戒")))

    print(model.most_similar("张无忌"))


    """

    print("")
    sentence1 = "密钥的保管在人工参与密钥管理情况下，应用系统的密钥和密钥备份原则上由业务部门保管。"
    sentence2 = "本规定所称密钥管理是指对密钥的生成、保管、分发、使用、备份和恢复、撤销和更新等生命周期的管理，防范密钥泄漏，保障信息数据的安全。"
    #print("计算机不能够计算 vs 计算机能够计算")
    sim = calcute_sentences_distance(sentence1, sentence2)
    print(sim)

    print("")
    with open("04.txt", mode="r", encoding="utf8") as f:
        text = f.read()
        summary = summarize(text, 3)
        print(summary)
    """


def find_sentences(text_one,text_two,n):
    """查找最相似的 n 个句子
    """

    def helper(text):
        tokens = cut_sentences(text)
        sentences = []
        #sents = []
        for sent in tokens:
            sentences.append(sent)
            #sents.append([word for word in jieba.cut(sent) if word])

        #sents = filter_symbols(sents)
        #sents = filter_model(sents)
        return sentences

    def most_similar_sentens(wendang_one,sentences_fagui,n):
        values = {}
        index = 0
        for fagui_one in sentences_fagui:
            value = calcute_sentences_distance(wendang_one,fagui_one)
            values[index] = value
            index += 1
        sorted_values = sorted(values.items(),key = lambda x:x[1],reverse = True)[0:n]
        return [(sentences_fagui[item[0]],item[1]) for item in sorted_values]


    sentences_wendang = helper(text_one)
    sentences_fagui = helper(text_two)

    for wendang_one in sentences_wendang:
        print(wendang_one)
        most_similars = most_similar_sentens(wendang_one,sentences_fagui,n)
        index = 0
        for similar in most_similars:
            print("  {}:{} {}".format(index,similar[0],similar[1]))
            index += 1
        print("\n\n")

    





if __name__ == '__main__':

    model = word2vec.Word2Vec.load("xdstar.lexicon")
    #test_model()


    with open("wendang2.txt", "r", encoding='utf-8') as text_one:
        text_one = text_one.read().replace('\n', '')
        with open("fagui.txt", "r", encoding='utf-8') as text_two:
            text_two = text_two.read().replace('\n', '')
            find_sentences(text_one,text_two,10)

        

