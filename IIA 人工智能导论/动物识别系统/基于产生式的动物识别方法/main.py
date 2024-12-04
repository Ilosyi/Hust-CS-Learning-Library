from PyQt5 import QtCore, QtGui, QtWidgets

# 定义规则集
RULES = [
    {"if": {"蹄": True, "长尾巴": True, "骑乘或拉车": True}, "then": "马"},
    {"if": {"羽毛": True, "喙": True, "下蛋": True}, "then": "鸡"},
    {"if": {"体型小": True, "柔软皮肤": True, "锋利牙齿和爪子": True}, "then": "猫"},
    {"if": {"体型多样": True, "毛发": True, "锋利牙齿": True, "人类伴侣": True}, "then": "狗"},
    {"if": {"毛发": True}, "then": "哺乳动物"},
    {"if": {"奶": True}, "then": "哺乳动物"},
    {"if": {"羽毛": True}, "then": "鸟"},
    {"if": {"会飞": True, "下蛋": True}, "then": "鸟"},
    {"if": {"吃肉": True}, "then": "食肉动物"},
    {"if": {"犬齿": True, "爪": True, "眼盯前方": True}, "then": "食肉动物"},
    {"if": {"哺乳动物": True, "蹄": True}, "then": "有蹄类动物"},
    {"if": {"哺乳动物": True, "反刍动物": True}, "then": "有蹄类动物"},
    {"if": {"哺乳动物": True, "食肉动物": True, "黄褐色": True, "暗斑点": True}, "then": "金钱豹"},
    {"if": {"哺乳动物": True, "食肉动物": True, "黄褐色": True, "黑色条纹": True}, "then": "虎"},
    {"if": {"有蹄类动物": True, "长脖子": True, "长腿": True, "暗斑点": True}, "then": "长颈鹿"},
    {"if": {"有蹄类动物": True, "黑色条纹": True}, "then": "斑马"},
    {"if": {"鸟": True, "长脖子": True, "长腿": True, "不会飞": True, "黑白二色": True}, "then": "鸵鸟"},
    {"if": {"鸟": True, "会游泳": True, "不会飞": True, "黑白二色": True}, "then": "企鹅"},
    {"if": {"鸟": True, "善飞": True}, "then": "信天翁"}
]

# 特征映射字典（根据RULES顺序排列）
FEATURES = [
    "蹄", "长尾巴", "骑乘或拉车",  # 规则 1
    "羽毛", "喙", "下蛋",  # 规则 2
    "体型小", "柔软皮肤", "锋利牙齿和爪子",  # 规则 3
    "体型多样", "毛发", "锋利牙齿", "人类伴侣",  # 规则 4
    # 规则 5
    "奶",  # 规则 6
    # 规则 7
    "会飞",# 规则 8
    "吃肉",  # 规则 9
    "犬齿", "爪", "眼盯前方",  # 规则 10
    "哺乳动物",  # 规则 11
    "反刍动物",  # 规则 12
    "食肉动物", "黄褐色", "暗斑点",  # 规则 13
    "黄褐色", "黑色条纹",  # 规则 14
    "长脖子", "长腿", "暗斑点",  # 规则 15
    "有蹄类动物", "黑色条纹",  # 规则 16
    "鸟", "长脖子", "长腿", "不会飞", "黑白二色",  # 规则 17
    "会游泳",  # 规则 18
    "善飞",  # 规则 19
]

class AnimalRecognitionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('动物识别系统')
        self.setGeometry(100, 100, 800, 600)

        # 创建一个水平分割器
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        splitter.setGeometry(0, 0, 800, 600)

        # 左侧 - 特征选择区域与推理过程、结果展示区域
        leftWidget = QtWidgets.QWidget(self)
        leftLayout = QtWidgets.QVBoxLayout(leftWidget)

        # 设置标签字体
        self.label = QtWidgets.QLabel("请选择动物特征：", self)
        self.label.setStyleSheet("font: 16pt;")
        leftLayout.addWidget(self.label)

        self.featureButtons = {}
        y_position = 0
        for feature in FEATURES:
            label = QtWidgets.QLabel(feature, self)
            label.setStyleSheet("font: 14pt;")  # 增加字体大小
            leftLayout.addWidget(label)

            # 创建切换按钮（默认为"否"）
            button = QtWidgets.QPushButton("否", self)
            button.setStyleSheet("font: 14pt;")
            button.setCheckable(True)
            button.setChecked(False)  # 默认为"否"
            button.clicked.connect(self.toggle_button_state)
            self.featureButtons[feature] = button
            leftLayout.addWidget(button)

        # 将左侧的选择框添加到滚动区域
        scroll_area = QtWidgets.QScrollArea(self)
        scroll_area.setWidget(leftWidget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        # 右侧 - 推理过程和结果展示区域
        rightWidget = QtWidgets.QWidget(self)
        rightLayout = QtWidgets.QVBoxLayout(rightWidget)

        # 识别按钮
        self.recognizeButton = QtWidgets.QPushButton('识别', self)
        self.recognizeButton.setStyleSheet("font: 14pt;")
        self.recognizeButton.clicked.connect(self.on_recognize)
        rightLayout.addWidget(self.recognizeButton)

        # 结果标签
        self.resultLabel = QtWidgets.QLabel('识别结果将在这里显示', self)
        self.resultLabel.setStyleSheet("font: 14pt; color: blue;")
        rightLayout.addWidget(self.resultLabel)

        # 推理过程标签
        self.procedureLabel = QtWidgets.QLabel('推理过程：', self)
        self.procedureLabel.setStyleSheet("font: 14pt;")
        rightLayout.addWidget(self.procedureLabel)

        self.procedureTextEdit = QtWidgets.QTextEdit(self)
        self.procedureTextEdit.setReadOnly(True)
        self.procedureTextEdit.setStyleSheet("font: 12pt;")
        rightLayout.addWidget(self.procedureTextEdit)

        # 向分割器添加左右布局的控件
        splitter.addWidget(scroll_area)
        splitter.addWidget(rightWidget)

        self.show()

    def toggle_button_state(self):
        # 切换按钮的状态
        sender = self.sender()
        if sender.isChecked():
            sender.setText("是")
        else:
            sender.setText("否")

    def on_recognize(self):
        selected_features = {}
        # 获取用户选择的特征
        for feature, button in self.featureButtons.items():
            selected_features[feature] = button.isChecked()

        # 显示推理过程
        animal, inference_process = self.infer_animal(selected_features)

        # 显示结果
        self.resultLabel.setText(f"识别出的动物是：{animal}")

        # 展示推理过程
        self.procedureTextEdit.clear()  # 清空文本框
        if inference_process:
            for step in inference_process:
                self.procedureTextEdit.append(step)
        else:
            self.procedureTextEdit.append("未匹配任何规则")

    def infer_animal(self, features):
        inference_process = []  # 用于记录推理过程
        matched_animal = None  # 用于保存匹配的动物

        # 遍历所有规则
        for rule in RULES:
            rule_conditions = rule["if"]  # 获取规则的条件部分

            # 检查每个条件是否都匹配
            # 如果特征不存在，则默认返回False
            if all(features.get(key, False) == value for key, value in rule_conditions.items()):
                inference_process.append(f"规则匹配: {rule_conditions} => 结果是 {rule['then']}")
                # 保存匹配到的动物
                matched_animal = rule["then"]

        # 如果找到了匹配的动物，返回匹配结果；否则返回"未知动物"
        if matched_animal:
            return matched_animal, inference_process
        else:
            return "未知动物", ["未匹配任何规则"]



if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ex = AnimalRecognitionApp()
    app.exec_()
