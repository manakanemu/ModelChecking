#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import re
import traceback
import networkx as nx
import matplotlib.pyplot as plt

translate_dic = [
    ['$CHANGES', '$1', 'countChanged(row,col)'],
    ['$FIXES', '$2', 'countFixed(row,col)'],
    ['and', '&&', 'and'],
    ['or', '||', 'or']
]


def translate(sentence, input: int, output: int)->str:
    for word in translate_dic:
        sentence = sentence.replace(word[input], word[output])
    return sentence


def get_word_from_bracket(sentence, flag=True):
    if flag:
        match = re.match(r'[^{]*{.*}[^}]*', sentence)
    else:
        match = re.match(r'[^{]*{.*?}[^}]*', sentence)
    if match is None:
        raise Exception('"' + str(sentence) + '" 括号语法错误')
        return (0, 0)
    else:
        return match.regs[0]


class Model:
    transition_matrix = None
    size_matrix = None
    name_matrix = None
    quick_index = None
    U = None

    def load_state_unit(self, filepath):
        init_matrix = eval(read_conf(filepath))
        name_matrix = []
        state_matrix = []

        # 标记可达矩阵属性
        for key in init_matrix:
            name_matrix.append(key)
            state_matrix.append(init_matrix[key])

        # 生成状态转移矩阵和标记矩阵状态
        state_transition_matrix = self.joint_state_unit(state_matrix)

        self.transition_matrix = state_transition_matrix
        self.size_matrix = self.get_size_matrix(state_matrix)
        self.name_matrix = name_matrix

    # 拼接状态单元

    def joint_state_unit(self, state_matrix: list) -> list:
        state_transition_matrix = []
        for key in range(len(state_matrix)):
            if len(state_transition_matrix) == 0:
                state_transition_matrix = np.array(state_matrix[key])
            else:
                matrixa = state_transition_matrix
                matrixb = np.array(state_matrix[key])
                sizea = matrixa.shape[0]
                sizeb = matrixb.shape[0]
                state_transition_matrix = None;
                for rowa in range(sizea):
                    for rowb in range(sizeb):
                        vr = np.dot(matrixa[rowa].reshape(sizea, 1),
                                    matrixb[rowb].reshape(1, sizeb)).reshape(1, sizea * sizeb)
                        if state_transition_matrix is None:
                            state_transition_matrix = vr
                        else:
                            state_transition_matrix = np.concatenate((state_transition_matrix, vr), axis=0)
        return state_transition_matrix

    # 获取矩阵规模
    def get_size_matrix(self, state_matrix):
        size_matrix = []
        for matrix in state_matrix:
            size_matrix.append(len(matrix))
        return size_matrix

    def load_state_limiter(self, filepath):
        state_limits = eval(read_conf(filepath))
        self.run_state_limiter(state_limits)

    # 根据状态限制器规范化状态转移矩阵
    def run_state_limiter(self, limits: list):
        size = self.transition_matrix.shape[0]
        for i in range(len(limits)):
            try:
                limit = limits[i]
                start, end = get_word_from_bracket(limit)
                limit = limit[start + 1:end - 1].strip()
                limit = translate(limit, 0, 1)
                for j in range(len(self.name_matrix)):
                    name = self.name_matrix[j]
                    limit = limit.replace(name, '_[' + str(j) + ']')
                limit = translate(limit, 1, 2)
                limits[i] = limit
            except:
                traceback.print_exc()
        for i in range(size):
            _ = self.get_coordinate(i)
            pass
            for limit in limits:
                try:
                    if eval(limit):
                        self.transition_matrix[i, :] = 0
                        self.transition_matrix[:, i] = 0
                except:
                    print(limit, ":语法错误")

    def load_transition_limiter(self, filepath):
        transition_limits = eval(read_conf(filepath))
        self.run_transition_limiter(transition_limits)
        self.gen_quick_index()

    # 根据转移限制器规范会状态转移矩阵

    def run_transition_limiter(self, limits):
        rows, cols = self.transition_matrix.shape

        for i in range(len(limits)):
            try:
                limit = limits[i].strip()
                start, end = get_word_from_bracket(limit)
                limit = limit[start + 1:end - 1]
                limit = translate(limit, 0, 1)
                for j in range(len(self.name_matrix)):
                    name = self.name_matrix[j]
                    limit = limit.replace(name + '.from', 'from_coordinate[' + str(j) + ']')
                    limit = limit.replace(name + '.to', 'to_coordinate[' + str(j) + ']')
                limit = translate(limit, 1, 2)
                limits[i] = limit
            except:
                traceback.print_exc()
        for row in range(rows):
            for col in range(cols):
                if self.transition_matrix[row, col] == 1:
                    from_coordinate = self.get_coordinate(row)
                    to_coordinate = self.get_coordinate(col)
                    for limit in limits:
                        if eval(limit):
                            self.transition_matrix[row, col] = 0
                            break

    # 根据矩阵行列获取connection坐标
    def get_coordinate(self, axis):
        coordinate = []
        for i in range(len(self.size_matrix)):
            if i == len(self.size_matrix) - 1:
                co = axis
            else:
                scale = 1
                for j in range(i + 1, len(self.size_matrix)):
                    scale *= self.size_matrix[j]
                co = axis // scale
            coordinate.append(co)
            axis -= co * scale
        return coordinate

    def to_state(self, coordinate):
        state = 0
        size_vector = np.array(self.size_matrix)
        for i in range(len(self.size_matrix) - 1):
            scale = 1
            for j in range(i + 1, len(self.size_matrix)):
                scale *= self.size_matrix[j]
            state += coordinate[i] * scale
        return state + coordinate[-1]

    def get_relation(self, transition):
        from_coordinate = self.get_coordinate(transition[0])
        to_coordinate = self.get_coordinate(transition[1])
        return from_coordinate, to_coordinate

    # 生成快速查找三元组
    def gen_quick_index(self):
        rows, cols = self.transition_matrix.shape
        quick_index = []
        for row in range(rows):
            for col in range(cols):
                if self.transition_matrix[row, col] == 1:
                    quick_index.append([row, col])

        self.quick_index = quick_index
        self.U = set([x[0] for x in self.quick_index] + [x[1] for x in self.quick_index])

    def show_transition_matrix(self):
        print('\n======状态转移矩阵======')
        print(self.transition_matrix)
        print('=======================\n')

    def show_size_matrix(self):
        print('\n========尺寸矩阵========')
        print(self.size_matrix)
        print('=======================\n')

    def show_name_matrix(self):
        print('\n=======状态名矩阵=======')
        print(self.name_matrix)
        print('=======================\n')

    def show_quick_index(self):
        print('\n======快速检索矩阵======')
        for q in self.quick_index:
            print(q)
        print('=======================\n')

    def show_U(self):
        print('\n======可达状态全集======')
        print(self.U)
        print('=======================\n')

    def get_named_edges(self):
        edges = []
        for e in self.quick_index:
            from_coordinate, to_coordinate = self.get_relation(e)
            from_name, to_name = '', ''
            for i in range(len(self.name_matrix)):
                name = self.name_matrix[i]
                from_name += name + str(from_coordinate[i])
                to_name += name + str(to_coordinate[i])
            edges.append([from_name, to_name])
        return edges

    def show_transition_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from(model.get_named_edges())
        plt.figure(3, figsize=(10, 10))
        nx.draw_networkx(graph, with_labels=True, node_size=1000,
                         node_color='#F596AA', edge_color='#58B2DC', arrowize=50)
        plt.show()



class Checker:
    queries = None
    init_state = None
    extension = []

    def load_init(self, filepath):
        self.init_state = eval(read_conf(filepath))

    def load_query(self, filepath):
        self.queries = eval(read_conf(filepath))

    def parser(self, sentence: str):
        sentence = sentence.strip()
        first = sentence[0]
        next = sentence[1:]
        if (first == 'A'):
            sa = A(self.parser(next))
            return sa
        if (first == 'E'):
            se = E(self.parser(next))
            return se
        if (first == 'X'):
            sx = X(self.parser(next))
            return sx
        if (first == 'F'):
            sf = F(self.parser(next))
            return sf
        if (first == 'N'):
            sn = N(self.parser(next))
            return sn
        if (first == 'G'):
            sn = G(self.parser(next))
            return sn
        if (first == '{'):
            try:
                start, end = get_word_from_bracket(sentence, False)
                query = sentence[start + 1:end - 1]
                sentence = sentence[end:]
                if not sentence:
                    sq = Q(query)
                    return sq
                else:
                    first = sentence[0]
                    next = sentence[1:]
                    if (first == '&'):
                        sand = AND(Q(query), self.parser(next))
                        return sand
                    if (first == '|'):
                        sor = OR(Q(query), self.parser(next))
                        return sor
                    if (first == '-'):
                        sto = TO(Q(query), self.parser(next))
                        return sto
                    if (first == 'U'):
                        su = U(Q(query), self.parser(next))
                        return su
            except:
                print('condition 语法错误')
        if (first == "("):
            bracket_len = bracket_matching(sentence)
            if bracket_len == len(sentence):
                return self.parser(sentence[1:-1])
            else:
                query = sentence[1:bracket_len]
                sentence = sentence[bracket_len:-1].strip()
                first = sentence[0]
                next = sentence[1:]
                if (first == '&'):
                    sand = AND(Q(query), self.parser(next))
                    return sand
                if (first == '|'):
                    sor = OR(Q(query), self.parser(next))
                    return sor
                if (first == '-'):
                    sto = TO(Q(query), self.parser(next))
                    return sto
                if (first == 'U'):
                    su = U(Q(query), self.parser(next))
                    return su

    def check(self):
        if self.init_state is None:
            raise Exception('执行check前需要先执行load_init_state函数')
        elif self.queries is None:
            raise Exception('执行check前需要先执行load_query')
        else:
            result = []
            for i in range(len(self.queries)):
                query = self.queries[i]
                self.extension.append(self.parser(query))
                if len(self.extension[i]) == 0:
                    result.append(False)
                    continue
                start, end = get_word_from_bracket(self.init_state)
                init = self.init_state[start + 1:end - 1]
                init = translate(init, 0, 1)
                for j in range(len(model.name_matrix)):
                    name = model.name_matrix[j]
                    init = init.replace(name, '_[' + str(i) + ']')
                init = translate(init, 1, 2)
                for ex in self.extension[i]:
                    _ = model.get_coordinate(ex)
                    if eval(init):
                        result.append(True)
                        break
                    else:
                        result.append(False)
                        break
            return result

    def show_query(self):
        print('\n======查询语句======')
        print(self.query)
        print('===================\n')

    def show_init_state(self):
        print('\n======初始状态======')
        print(self.init_state)
        print('===================\n')

    def show_extension(self):
        print('\n========外延========')
        print(self.extension)
        print('====================\n')


# 读取配置
def read_conf(file_path):
    with open(file_path, 'r') as file:
        text = file.read().replace('\n', '')
    return text


# CHANGED保留字实体
def countChanged(row, col):
    from_coordinate = model.get_coordinate(row)
    to_coordinate = model.get_coordinate(col)
    count = 0
    for i in range(len(model.size_matrix)):
        if from_coordinate[i] != to_coordinate[i]:
            count += 1
    return count


# FIXED保留字实体
def countFixed(row, col):
    return len(model.size_matrix) - countChanged(row, col)


# 原子命题

def Q(state: str):
    for i in range(len(model.name_matrix)):
        name = model.name_matrix[i]
        state = state.replace(name + ' ', 'to_coordinate[' + str(i) + '] ')

    extension = set()
    for tran in model.quick_index:
        from_coordinate, to_coordinate = model.get_relation(tran)
        if eval(state):
            extension.add(tran[1])
    return extension


# 非运算符
def N(extension: set):
    return model.U - extension


# 与运算符
def AND(extension1: set, extension2: set):
    return extension1 & extension2


# 或运算符
def OR(extension1: set, extension2: set):
    return extension1 | extension2


# 蕴含运算符
def TO(extension1: set, extension2):
    return OR(N(extension1), extension2)


# next运算符
def X(extension: set):
    new_set = set()
    for tran in model.quick_index:
        if tran[1] in extension:
            new_set.add(tran[0])
    return new_set


# finally运算符
def F(extension):
    while True:
        new_extension = X(extension) | extension
        if new_extension == extension:
            return new_extension
        else:
            extension = new_extension


# global运算符
def G(extension: set):
    return N(F(N(extension)))


# until运算符
def U(extension1: set, extension2: set):
    while True:
        new_set = extension2.copy()
        for tran in model.quick_index:
            if tran[1] in new_set and tran[0] in extension1:
                new_set.add(tran[0])
        if new_set == extension2:
            return new_set
        else:
            extension2 = new_set


# all运算符
def A(extension):
    new_set = model.U.copy()
    for tran in model.quick_index:
        if tran[1] not in extension and tran[0] in new_set:
            new_set.remove(tran[0])
    new_set = new_set & extension
    return new_set


# exist运算符
def E(extension):
    new_set = N(A(N(extension)))
    return new_set


# 括号匹配
def bracket_matching(sentence: str):
    brackets = []
    for i in range(len(sentence)):
        if sentence[i] == '(':
            brackets.append('(')
        if (sentence[i] == ')'):
            brackets.pop()
            if len(brackets) == 0:
                return i + 1
    if len(brackets) == 0:
        return len(sentence)
    else:
        raise Exception('括号语法错误')


# 获取初始可达矩阵
model = Model()
model.load_state_unit('connection_matrix.river.json')
model.show_transition_matrix()

# 使用状态限制器规范规范状态转移矩阵
model.load_state_limiter('state_limit.river.json')
model.show_transition_matrix()

# 使用转移限制器规范状态转移矩阵
model.load_transition_limiter('transition_limit.river.json')
model.show_transition_matrix()

checker = Checker()
checker.load_init('init.river.json')
checker.load_query('condition.river.json')
print(checker.check())
model.show_transition_graph()


#
# checker.show_query()
# checker.show_init_state()
# checker.show_extension()
#
# model.show_transition_graph()
#
# # 语法分析器
#

