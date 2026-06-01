import os
import re
import base64
import html
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# 1. 基础配置
# =========================

load_dotenv()


def get_secret_or_env(key: str, default=None):
    """
    优先读取 Streamlit Cloud 的 Secrets；
    如果没有，则读取本地 .env；
    如果都没有，则使用默认值。
    """
    try:
        value = st.secrets.get(key, None)
    except Exception:
        value = None

    if not value:
        value = os.getenv(key, default)

    return value


DEEPSEEK_API_KEY = get_secret_or_env("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    st.error("没有检测到 DEEPSEEK_API_KEY，请先在 .env 文件或 Streamlit Secrets 中配置。")
    st.stop()

client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# 图片文字识别 OCR 配置：使用阿里云百炼 DashScope 的 OpenAI 兼容接口
DASHSCOPE_API_KEY = get_secret_or_env("DASHSCOPE_API_KEY")
QWEN_OCR_MODEL = get_secret_or_env("QWEN_OCR_MODEL", "qwen-vl-ocr")

ocr_client = None
if DASHSCOPE_API_KEY:
    ocr_client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

st.set_page_config(
    page_title="图形化编程学习助手",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# 1.5 页面样式
# =========================

def inject_custom_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 14% 12%, rgba(255, 210, 117, 0.18) 0, rgba(255, 210, 117, 0) 26%),
                radial-gradient(circle at 86% 8%, rgba(112, 169, 255, 0.16) 0, rgba(112, 169, 255, 0) 28%),
                linear-gradient(180deg, #F7FBFF 0%, #EEF6FF 48%, #F8FBFF 100%);
        }

        .block-container {
            max-width: 1240px;
            padding-top: 2.6rem;
            padding-bottom: 6rem;
        }

        section[data-testid="stSidebar"] {
            background: #FFFFFF;
            border-right: 1px solid #E6EDF5;
        }

        button[data-testid="stSidebarCollapseButton"],
        div[data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .main-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #E3ECF6;
            border-radius: 20px;
            padding: 17px 22px;
            box-shadow: 0 12px 28px rgba(52, 86, 130, 0.08);
            margin-bottom: 16px;
        }

        .title-row {
            display: flex;
            align-items: center;
            gap: 14px;
        }

        .title-icon {
            width: 48px;
            height: 48px;
            border-radius: 15px;
            background: linear-gradient(135deg, #FFF2D6 0%, #EAF4FF 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.65rem;
            box-shadow: inset 0 0 0 1px rgba(224, 235, 248, 0.9);
            flex: 0 0 48px;
        }

        .title-text {
            min-width: 0;
        }

        .app-title {
            font-size: 1.82rem;
            line-height: 1.16;
            font-weight: 850;
            color: #273247;
            margin: 0 0 5px 0;
            letter-spacing: -0.02em;
        }

        .app-desc {
            font-size: 0.92rem;
            color: #66758F;
            margin: 0;
        }

        div[data-testid="stTextInput"] input {
            border-radius: 14px;
            border: none !important;
            background: #FFFFFF;
            height: 42px;
            box-shadow: none !important;
            outline: none !important;
        }

        div[data-testid="stTextInput"] [data-baseweb="input"] {
            border-radius: 14px !important;
            border: 1px solid #DDE8F4 !important;
            background: #FFFFFF !important;
            box-shadow: none !important;
        }

        div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
            border: 1px solid #80B7FF !important;
            box-shadow: 0 0 0 2px rgba(128, 183, 255, 0.12) !important;
        }

        div[data-testid="stTextInput"] input:focus {
            outline: none !important;
            box-shadow: none !important;
            border: none !important;
        }

        div[data-testid="stTextInput"] label {
            color: #263449;
            font-weight: 600;
        }

        .starter-area {
            margin: 2px 0 14px 38px;
            max-width: 500px;
        }

        .starter-area .stButton {
            margin-bottom: 4px;
        }

        .starter-area .stButton > button {
            width: fit-content;
            max-width: 100%;
            min-height: 28px;
            border-radius: 12px;
            border-top-left-radius: 5px;
            border: 1px solid #E5EDF6;
            background: rgba(255, 255, 255, 0.92);
            color: #44536A;
            padding: 0.24rem 0.50rem;
            text-align: left;
            font-size: 0.78rem;
            box-shadow: 0 4px 10px rgba(52, 86, 130, 0.035);
        }

        .starter-area .stButton > button:hover {
            border-color: #9BC7FF;
            color: #1F5FBF;
            background: #FFFFFF;
            transform: translateY(-1px);
        }

        .stButton > button {
            border-radius: 15px;
            border: 1px solid #D8E5F4;
            background: rgba(255,255,255,0.94);
            color: #3E5069;
            padding: 0.48rem 0.74rem;
            min-height: 40px;
            font-size: 0.88rem;
            box-shadow: 0 7px 16px rgba(52, 86, 130, 0.055);
            transition: all 0.16s ease-in-out;
            text-align: left;
        }

        .stButton > button:hover {
            border-color: #8EBEFF;
            color: #1F5FBF;
            background: #FFFFFF;
            transform: translateY(-1px);
        }

        .chat-wrap {
            display: flex;
            width: 100%;
            margin: 10px 0;
            align-items: flex-start;
            gap: 8px;
        }

        .chat-wrap.user {
            justify-content: flex-end;
        }

        .chat-wrap.assistant {
            justify-content: flex-start;
        }

        .chat-content {
            min-width: 0;
            display: flex;
        }

        .chat-wrap.user .chat-content {
            max-width: 78%;
            justify-content: flex-end;
        }

        .chat-wrap.assistant .chat-content {
            max-width: 88%;
            justify-content: flex-start;
        }

        .chat-bubble {
            display: inline-block;
            width: fit-content;
            max-width: 100%;
            padding: 10px 13px;
            border-radius: 16px;
            font-size: 0.95rem;
            line-height: 1.55;
            white-space: pre-wrap;
            word-break: break-word;
            box-sizing: border-box;
            box-shadow: 0 6px 18px rgba(52, 86, 130, 0.06);
        }

        .chat-bubble.user {
            background: #B9ECA0;
            color: #1F2B1D;
            border-top-right-radius: 5px;
        }

        .chat-bubble.assistant {
            background: #FFFFFF;
            color: #273247;
            border: 1px solid #E5EDF6;
            border-top-left-radius: 5px;
        }

        .chat-bubble .md-line {
            margin: 0.24rem 0;
        }

        .chat-bubble .md-heading {
            font-weight: 850;
            font-size: 1.08rem;
            color: #21304A;
            margin: 0.72rem 0 0.36rem 0;
        }

        .chat-bubble .md-heading:first-child {
            margin-top: 0;
        }

        .chat-bubble .md-number,
        .chat-bubble .md-bullet {
            margin: 0.28rem 0;
        }

        .chat-bubble .md-space {
            height: 0.45rem;
        }

        .chat-bubble strong {
            color: #1F2B3F;
            font-weight: 850;
        }

        .chat-avatar {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 0.9rem;
            flex: 0 0 30px;
            margin-top: 2px;
            box-shadow: 0 4px 12px rgba(52, 86, 130, 0.08);
        }

        .chat-avatar.user {
            background: #DFF6D3;
            color: #2F6B27;
            order: 2;
        }

        .chat-avatar.assistant {
            background: #FFFFFF;
            color: #273247;
            border: 1px solid #E5EDF6;
        }

        div[data-testid="stChatInput"] {
            background: rgba(244, 248, 252, 0.88);
        }

        div[data-testid="stChatInput"] > div {
            border: 1px solid #CFE0F5 !important;
            box-shadow: 0 8px 22px rgba(52, 86, 130, 0.07) !important;
        }

        div[data-testid="stChatInput"]:focus-within > div {
            border: 1px solid #80B7FF !important;
            box-shadow: 0 0 0 2px rgba(128, 183, 255, 0.16) !important;
        }

        div[data-testid="stChatInput"] [data-baseweb="textarea"],
        div[data-testid="stChatInput"] textarea {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
            background: transparent !important;
        }

        div[data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within,
        div[data-testid="stChatInput"] textarea:focus {
            border: none !important;
            box-shadow: none !important;
            outline: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


inject_custom_css()


# =========================
# 2. 系统提示词
# =========================

SYSTEM_PROMPT = """
你是“图形化编程学习助手”，服务于小学高年级 Scratch 图形化编程课堂。

你的服务对象包括教师和学生。

总原则：
1. 教师端重在支持教学设计、课堂调控、任务单生成、问题链设计、调试提示、展示评价和教学反思。
2. 学生端重在引导思考，不能直接替学生完成完整程序，不能直接给出完整答案。
3. 所有回答必须围绕小学图形化编程任务展开。
4. 面向学生时，要用短句、分步提示，每次只推进一点。
5. 面向教师时，要规范、清晰、可操作，贴近小学课堂实际。
6. 图形化编程教学流程为：情境创设、任务分析、实践创作、展示评价。
7. 任务分析建议采用：作品 → 角色 → 动作、规则、效果；必要时可引导教师和学生借助思维导图、流程图或任务分析表梳理作品结构与程序逻辑。
8. 学生调试时优先提示检查：启动事件、角色脚本归属、重复执行、条件是否满足、变量是否初始化、角色是否隐藏、造型、位置、方向是否设置正确。
9. 智能体始终遵循：教师主导，学生主体，智能辅助。
10. 不生成脱离图形化编程课堂实际的空泛内容。
11. 回答中不得出现 <br>、<p>、<div> 等 HTML 标签。
12. 不使用 Markdown 表格，尤其是教学过程不得使用“|”组织表格。
13. 教学过程采用“小标题 + 分点说明”的形式呈现，避免表格错位。
14. 如需分条，请使用“①②③”、分号或短句表达。
15. 不要输出 HTML 标签，不要输出网页代码。
"""


def build_role_prompt(role: str) -> str:
    if role == "教师端":
        return """
当前用户身份：教师。

你是小学高年级 Scratch 图形化编程教学支持助手。

请根据教师输入，自动判断其需求属于以下哪一类：
1. 教学设计；
2. 任务单设计；
3. 任务分析模板；
4. 课堂问题链；
5. 调试提示；
6. 展示评价；
7. 教学反思；
8. 智能体使用设计。

回答要求：

一、总体要求
1. 内容必须贴近小学高年级 Scratch 图形化编程课堂。
2. 不写空泛套话，要具体、清晰、可操作。
3. 教学流程优先采用：情境创设 → 任务分析 → 实践创作 → 展示评价。
4. 任务分析统一采用“作品 → 角色 → 动作、规则、效果”的方式；在任务分析环节应鼓励教师使用思维导图、流程图或任务分析表帮助学生梳理作品结构、角色关系和程序逻辑。
5. 不得使用 Markdown 表格，不得使用“|”组织表格。
6. 不得出现 <br>、<p>、<div> 等 HTML 标签。
7. 教学过程请使用“小标题 + 分点说明”的形式呈现，每个环节分别写清教师活动、学生活动、智能体支持和设计意图。
8. 语言要规范，适合教师直接修改后用于论文、教案或课堂材料。

二、当教师要求生成教学设计时，必须按照以下结构输出：

（一）教学设计基础

必须包括以下内容：

1. 教学对象
说明适用年级、学生已有基础和学习背景。

2. 学情分析
从以下三个方面展开：
① 认知基础：学生对角色、舞台、积木、程序逻辑等已有理解。
② 技能基础：学生是否具备基本拖拽积木、运行程序、修改脚本等操作经验。
③ 学习特点：学生可能存在的兴趣特点、操作差异、任务理解困难或调试困难。

3. 教学目标
按照义务教育信息科技课程标准中的四个核心素养维度表述：
① 信息意识；
② 计算思维；
③ 数字化学习与创新；
④ 信息社会责任。

目标要结合具体 Scratch 作品主题，不要空泛。

4. 教学重难点
① 教学重点：本节课学生必须掌握的核心知识、关键积木或程序结构。
② 教学难点：学生在任务分析、程序搭建或调试优化中可能遇到的主要困难。

5. 教学策略
重点说明突破重难点的策略，例如：
① 通过作品展示引出任务，激发学生创作兴趣；
② 采用“作品 → 角色 → 动作、规则、效果”的方式进行任务拆解；
③ 教师通过板书、投屏或任务单提供思维导图框架，引导学生梳理作品结构、角色关系和程序逻辑；
④ 学生在思路不清时，可借助智能体整理作品逻辑、生成思维导图框架或完善任务分析表，但最终分析结果应由学生自行修改和完善；
⑤ 通过教师关键示范和学生实践结合促进理解；
⑥ 通过巡视指导、同伴交流和共性问题讲解帮助学生调试修改。

6. 教学环境与资源
包括图形化编程平台、计算机或平板设备、作品素材、任务单、思维导图、流程图、评价表、智能体等。

（二）教学过程

教学过程不得使用表格，必须按照以下固定格式输出：

#### 1. 情境创设
- 教师活动：
- 学生活动：
- 智能体支持：
- 设计意图：

#### 2. 任务分析
- 教师活动：
- 学生活动：
- 智能体支持：
- 设计意图：

#### 3. 实践创作
- 教师活动：
- 学生活动：
- 智能体支持：
- 设计意图：

#### 4. 展示评价
- 教师活动：
- 学生活动：
- 智能体支持：
- 设计意图：

（三）教学反思

教学反思必须作为教学设计结尾，简要包括以下内容：

1. 目标达成反思：说明本节课教学目标是否基本达成。
2. 学生学习反思：说明学生在任务理解、程序搭建、调试修改中的表现。
3. 智能体使用反思：说明智能体在哪些环节提供了支持，是否存在使用不足。
4. 改进建议：提出后续教学中可以优化的地方。

三、关于“智能体支持”的要求

1. 智能体支持必须根据具体课的内容和环节合理填写，不得机械套用。
2. 情境创设环节一般不安排学生直接使用智能体。可以写“教师课前借助智能体生成导入问题或旧知唤醒问题”，也可以写“本环节不直接使用智能体”。
3. 任务分析环节可以体现教师使用思维导图引导全班分析任务，学生也可以向智能体描述自己的作品想法，由智能体帮助梳理“作品 → 角色 → 动作、规则、效果”、角色关系、功能模块和程序运行逻辑，并形成可用于完善思维导图或任务分析表的提示。
4. 实践创作环节是智能体支持的重点，可以体现学生在程序调试、积木提示、错误排查、作品优化中向智能体获取分步提示。
5. 展示评价环节不得写“智能体对作品进行评分”。智能体不能替代教师进行正式评价。
6. 展示评价环节可以写“教师课前借助智能体生成评价问题”或“学生可借助智能体整理展示表达思路”，但正式评价仍由教师依据评价量表完成。
7. 如果某一环节不适合使用智能体，可以明确写“本环节不直接使用智能体”。

四、当教师要求生成任务单时

尽量包含：
1. 作品名称；
2. 任务分析；
3. 基础任务；
4. 提升任务；
5. 拓展任务；
6. 自我检查。

任务单也不要使用 Markdown 表格，可以用标题和分点呈现。

五、当教师要求生成调试提示时

要适合教师课堂讲解和巡视指导，围绕学生常见问题展开，例如：
1. 角色不动；
2. 角色只动一次；
3. 分数不增加；
4. 变量没有初始化；
5. 程序写错角色；
6. 条件判断没有触发；
7. 角色隐藏后没有显示；
8. 碰到边缘后角色方向异常。

六、当教师要求生成评价建议时

可以围绕完整性、技术性、创新性和艺术性展开，但不得让智能体替代教师评分。

七、如果涉及实验班和对照班

需说明：
1. 实验班使用智能体支持；
2. 对照班采用常规教学支持；
3. 两班在教学主题、课时安排、学习任务和作品要求上保持一致。
"""
    else:
        return """
当前用户身份：学生。

你是小学 Scratch 图形化编程学习助手。

请根据学生输入，自动判断其问题属于以下哪一类：
1. 任务理解；
2. 任务分析；
3. 积木功能理解；
4. 程序实现提示；
5. 程序调试；
6. 作品优化；
7. 展示表达。

学生端回答规则：

一、基本规则
1. 不能直接给完整程序。
2. 不能直接给完整积木组合。
3. 不能直接替学生完成答案。
4. 要用短句，适合小学五年级学生理解。
5. 如果信息不足，最多追问2个关键问题。
6. 如果是调试问题，先给1—2个最可能的检查方向。
7. 如果是任务分析，采用“作品 → 角色 → 动作、规则、效果”的方式引导；如果学生说“帮我画思维导图”“帮我梳理逻辑”“我不知道怎么分析作品”，应帮助学生整理思维导图框架，但不能直接替学生完成全部内容。
8. 如果是作品优化，只给1—2个可以自己尝试的建议。
9. 要鼓励学生先尝试、运行、观察效果，再继续修改。
10. 不使用 Markdown 表格，不得出现 <br>、<p>、<div> 等 HTML 标签。

二、关于“怎么做某个作品”的回答规则

1. 当学生问“怎么做某个作品”“这个游戏怎么做”“帮我做某个作品”时，默认判断为任务分析类问题。
2. 第一次回答只能帮助学生分析任务，不得直接给出具体积木名称、积木连接顺序或完整操作步骤。
3. 第一次回答必须围绕“作品 → 角色 → 动作、规则、效果”展开。
4. 第一次回答最后只能追问1—2个问题，引导学生明确最基础的小功能。
5. 只有当学生已经明确目标功能，或者说明“我已经用了哪些积木、现在出现什么问题”时，才可以提示关键积木类别。
6. 即使提示积木，也不能一次性给出完整程序顺序，只能给1—2个提示或检查方向。
7. 不得使用“第一步拖出……第二步拖出……第三步添加……”这种完整操作式回答。
8. 对基础任务，应先引导学生思考“动作是否需要重复”“触发条件是什么”“角色碰到边缘后应怎样变化”，再根据学生回答继续推进。

三、关于课堂表现评价

1. 当学生要求“评价我这节课的表现”“评价我的课堂表现”“我这节课表现怎么样”时，不得直接判断学生表现好坏，因为你无法完整观察学生课堂行为。
2. 应提供统一的自我评价框架，引导学生从任务理解、任务分析、程序搭建、调试修改、合作交流、作品优化等方面进行回顾。
3. 可以用分点方式呈现评价维度，但不得直接给学生打分。
4. 不得替代教师进行正式评价。
5. 如果学生进一步说明自己完成了哪些功能、遇到了什么问题、如何解决问题，可以帮助学生整理一段学习表现小结，并提出1—2条具体改进建议。
6. 回答时应避免直接说“你表现很好”“你完成得很棒”等缺乏依据的判断。

四、关于 Scratch 功能解释

1. 必须区分“Scratch 内置积木可以直接实现”和“需要额外程序实现”的情况。
2. 不得把不同效果混为一谈。
3. 不得主动编造多个复杂实现方案。
4. 如果确实需要提供多个方案，必须说明哪个是基础方案，哪个是进阶方案。
5. 面向初学者时，应优先推荐课堂中最基础、最稳定的实现方式，不要一开始引入过难的坐标判断、广播、克隆等内容。

五、关于边缘处理的解释

1. “碰到边缘就反弹”是 Scratch 内置积木可以直接实现的基础功能。
2. “从一端出去，从另一端出现”可以实现，但不是一个现成积木，需要用坐标判断，例如判断 x 坐标是否超过舞台边界。
3. “碰到边缘后掉头”在基础作品中通常可以用“碰到边缘就反弹”加“将旋转方式设为左右翻转”来实现。
4. 不要把“碰到边缘就反弹”和“穿屏出现”说成同一个功能。
5. 对第一节或基础任务，不主动推荐“穿屏出现”这种进阶效果。

六、关于某个主题

如果学生问“xx怎么做”，应优先引导其完成基础版。

不得一开始就把任务扩展成复杂游戏，不得主动生成过多复杂规则。

七、关于思维导图和逻辑梳理

1. 当学生说“帮我画思维导图”“帮我梳理逻辑”“我不知道怎么分析作品”时，应按照“作品 → 角色 → 动作、规则、效果”的结构帮助学生整理。
2. 可以给出思维导图的文字框架，例如：
作品名称：
主要角色：
每个角色的动作：
游戏规则：
预期效果：
可能用到的积木类别：
3. 不要直接替学生完成全部内容，应通过提问引导学生补充自己的想法。
4. 如果学生只说了作品名称，应先追问：作品里有哪些角色？每个角色要做什么？
5. 如果学生已经说明角色和功能，可以帮助学生检查逻辑是否完整，并提醒他把内容画到自己的思维导图中。
"""


# =========================
# 3. 日志保存
# =========================

def save_log(role: str, user_input: str, answer: str, student_name="", group_no="", topic=""):
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/chat_logs.csv"

    new_row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "student_name": student_name,
        "group_no": group_no,
        "topic": topic,
        "user_input": user_input,
        "answer": answer
    }

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(log_path, index=False, encoding="utf-8-sig")


def load_logs():
    log_path = "logs/chat_logs.csv"
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    return pd.DataFrame(
        columns=["time", "role", "student_name", "group_no", "topic", "user_input", "answer"]
    )


# =========================
# 4. 清理模型回复
# =========================

def clean_answer(text: str) -> str:
    """
    清理模型回复：
    1. 删除或替换 HTML 标签；
    2. 删除 Markdown 表格竖线，避免 Streamlit 渲染错位；
    3. 清理多余空行。
    """
    if not text:
        return ""

    # 处理常见 HTML 标签
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</div\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<div\s*>", "", text, flags=re.IGNORECASE)
    text = text.replace("&nbsp;", " ")

    # 移除其他可能残留的 HTML 标签
    text = re.sub(r"</?[^>]+>", "", text)

    # 防止 Markdown 表格符号导致页面错位
    text = text.replace("|", " ")

    # 清理 Markdown 表格分隔线，例如 --- --- ---
    text = re.sub(r"^\s*[-:]{3,}(\s+[-:]{3,})+\s*$", "", text, flags=re.MULTILINE)

    # 清理多余空格和空行
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_student_numbering(text: str) -> str:
    """
    只用于学生端：把常见的 1.、（1）、(1) 等编号统一成 ①②③。
    教师端不使用，避免影响教案中的正式层级编号。
    """
    if not text:
        return ""

    number_map = [
        ("10", "⑩"),
        ("1", "①"),
        ("2", "②"),
        ("3", "③"),
        ("4", "④"),
        ("5", "⑤"),
        ("6", "⑥"),
        ("7", "⑦"),
        ("8", "⑧"),
        ("9", "⑨"),
    ]

    for num, circled in number_map:
        text = text.replace(f"（{num}）", circled)
        text = text.replace(f"({num})", circled)

    converted_lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        prefix = line[:len(line) - len(stripped)]
        new_line = line

        for num, circled in number_map:
            for mark in [".", "．", "、"]:
                marker = f"{num}{mark}"
                if stripped.startswith(marker):
                    new_line = prefix + circled + " " + stripped[len(marker):].lstrip()
                    break
            if new_line != line:
                break

        converted_lines.append(new_line)

    return chr(10).join(converted_lines)


def apply_simple_bold(text: str) -> str:
    parts = text.split("**")
    if len(parts) < 3:
        return text

    result = []
    for index, part in enumerate(parts):
        if index % 2 == 1:
            result.append("<strong>" + part + "</strong>")
        else:
            result.append(part)

    return "".join(result)


def format_message_content(content: str) -> str:
    """
    将模型文本转换成更适合气泡显示的 HTML。
    主要处理教师端中常见的标题、加粗、分点和空行，让教学设计排版更整齐。
    """
    escaped = html.escape(content or "")
    escaped = apply_simple_bold(escaped)

    circled_numbers = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
    parts = []

    for line in escaped.splitlines():
        stripped = line.strip()

        if not stripped:
            parts.append('<div class="md-space"></div>')
            continue

        if stripped.startswith("#"):
            heading_text = stripped.lstrip("#").strip()
            if heading_text:
                parts.append('<div class="md-heading">' + heading_text + '</div>')
                continue

        if stripped.startswith("- "):
            parts.append('<div class="md-bullet">• ' + stripped[2:].strip() + '</div>')
            continue

        if stripped.startswith("• "):
            parts.append('<div class="md-bullet">' + stripped + '</div>')
            continue

        is_number_line = False
        for number_text in circled_numbers:
            if stripped.startswith(number_text):
                is_number_line = True
                break

        for number in range(1, 21):
            if stripped.startswith(str(number) + ".") or stripped.startswith(str(number) + "．") or stripped.startswith(str(number) + "、"):
                is_number_line = True
                break

        if stripped.startswith("（") or stripped.startswith("("):
            is_number_line = True

        if is_number_line:
            parts.append('<div class="md-number">' + stripped + '</div>')
        else:
            parts.append('<div class="md-line">' + stripped + '</div>')

    return "".join(parts)


def render_chat_bubble(role: str, content: str):
    """
    使用自定义 HTML 渲染类似微信的左右聊天气泡。
    用户问题显示在右侧，智能体回答显示在左侧。
    """
    safe_content = format_message_content(content or "")
    if role == "user":
        bubble_role = "user"
        avatar = "我"
    else:
        bubble_role = "assistant"
        avatar = "🐱"

    st.markdown(
        f"""
        <div class="chat-wrap {bubble_role}">
            <div class="chat-avatar {bubble_role}">{avatar}</div>
            <div class="chat-content">
                <div class="chat-bubble {bubble_role}">{safe_content}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================
# 4.5 图片文字识别 OCR
# =========================

def image_file_to_data_url(uploaded_file):
    """
    把上传的图片转换为 base64 格式，供 OCR 模型读取。
    """
    uploaded_file.seek(0)
    mime_type = uploaded_file.type or "image/png"
    image_bytes = uploaded_file.getvalue()
    encoded_image = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def recognize_image_text(uploaded_file):
    """
    使用 Qwen-OCR 识别图片中的文字。
    """
    if ocr_client is None:
        return "图片文字识别功能还没有配置 DASHSCOPE_API_KEY，请先在 Streamlit Secrets 中添加 DASHSCOPE_API_KEY。"

    image_url = image_file_to_data_url(uploaded_file)

    response = ocr_client.chat.completions.create(
        model=QWEN_OCR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "请提取这张图片中的全部可见文字。尽量保持原有顺序。不要解释，不要扩展，只输出识别到的文字。"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        temperature=0,
        max_tokens=1500
    )

    raw_text = response.choices[0].message.content
    return clean_answer(raw_text)


# =========================
# 5. 调用 DeepSeek
# =========================

def call_deepseek(role: str, messages: list) -> str:
    role_prompt = build_role_prompt(role)

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + role_prompt}
    ]

    for msg in messages:
        api_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    max_tokens = 3500 if role == "教师端" else 1000

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=api_messages,
        temperature=0.3,
        max_tokens=max_tokens
    )

    raw_answer = response.choices[0].message.content
    answer = clean_answer(raw_answer)

    if role == "学生端":
        answer = normalize_student_numbering(answer)

    return answer


# =========================
# 5.5 开场白问题
# =========================

def get_starter_questions(role: str):
    if role == "教师端":
        return [
            ("教学设计", "请帮我设计一节《打地鼠》Scratch图形化编程教学设计。"),
            ("任务单", "请帮我生成一份适合五年级学生的Scratch任务单。"),
            ("问题链", "请帮我设计一组课堂问题链。"),
            ("调试支持", "请帮我整理学生常见的Scratch调试问题。")
        ]

    return [
        ("任务分析", "我想做一个打地鼠小游戏，应该先分析哪些内容？"),
        ("调试帮助", "我的角色只动了一次，应该先检查哪里？"),
        ("变量问题", "我想让分数增加，但分数没有变化，可以怎么排查？"),
        ("图片识别", "我上传了一张程序截图，请帮我识别里面的文字并给我提示。")
    ]


def set_quick_prompt(prompt_text: str):
    st.session_state.quick_prompt = prompt_text


# =========================
# 6. 侧边栏：身份入口
# =========================

st.sidebar.title("使用入口")

user_role = st.sidebar.radio(
    "请选择使用入口",
    ["学生端", "教师端"]
)


TEACHER_PASSWORD = get_secret_or_env("TEACHER_PASSWORD", "teacher123")

if user_role == "教师端":
    password = st.sidebar.text_input("请输入教师端密码", type="password")
    if password != TEACHER_PASSWORD:
        st.warning("教师端需要密码。学生请使用左侧的“学生端”。")
        st.stop()


# 身份切换时清空当前页面对话
if "current_role" not in st.session_state:
    st.session_state.current_role = user_role

if st.session_state.current_role != user_role:
    st.session_state.current_role = user_role
    st.session_state.messages = []
    st.session_state.quick_prompt = ""


# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []


# 初始化学生会话标识
if "student_session_id" not in st.session_state:
    st.session_state.student_session_id = ""


# 初始化快捷问题
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = ""


# =========================
# 7. 页面主体
# =========================

st.markdown(
    """
    <div class="main-card">
        <div class="title-row">
            <div class="title-icon">🐱</div>
            <div class="title-text">
                <div class="app-title">图形化编程学习助手</div>
                <p class="app-desc">提问时说清楚：想实现什么、做到了哪一步、遇到了什么问题。</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

if user_role == "学生端":
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        student_name = st.text_input("姓名", placeholder="请输入姓名")

    with col2:
        group_no = st.text_input("小组号", placeholder="如：第1组")

    with col3:
        topic = st.text_input("当前主题", placeholder="如：打地鼠")

    # 只根据“小组号 + 当前主题”判断是否开启新对话
    # 只有小组号和当前主题都填写后，才进行判断
    # 小组号或当前主题任意一个发生变化，就清空当前页面对话
    # 注意：这不会删除已经保存到 logs/chat_logs.csv 的历史记录
    current_student_session = f"{group_no.strip()}_{topic.strip()}"

    if group_no.strip() and topic.strip():
        if current_student_session != st.session_state.student_session_id:
            st.session_state.student_session_id = current_student_session
            st.session_state.messages = []
            st.session_state.quick_prompt = ""
            st.rerun()

else:
    student_name = ""
    group_no = ""
    topic = ""


# =========================
# 7.5 开场白问题区域
# =========================

if not st.session_state.messages:
    render_chat_bubble(
        "assistant",
        "你好，我可以帮你分析 Scratch 任务、检查程序问题，也可以识别你上传的程序截图。你可以直接提问，也可以从下面选一个问题开始。"
    )

    st.markdown('<div class="starter-area">', unsafe_allow_html=True)
    starter_questions = get_starter_questions(user_role)

    for i, item in enumerate(starter_questions):
        label, question = item
        st.button(
            question,
            key=f"starter_{user_role}_{i}",
            on_click=set_quick_prompt,
            args=(question,)
        )
    st.markdown('</div>', unsafe_allow_html=True)


# =========================
# 8. 显示历史聊天消息
# =========================

for msg in st.session_state.messages:
    render_chat_bubble(msg["role"], msg["content"])


# =========================
# 9. 聊天输入框：支持文字 + 图片附件
# =========================

if user_role == "学生端":
    input_placeholder = "请输入问题，或点击附件上传图片。"
else:
    input_placeholder = "请输入教学需求，或点击附件上传图片。"

prompt = st.chat_input(
    input_placeholder,
    accept_file=True,
    file_type=["png", "jpg", "jpeg"]
)

if prompt:
    # 新版 st.chat_input 在 accept_file=True 时返回类似对象，包含 text 和 files
    user_input = getattr(prompt, "text", "") or ""
    uploaded_image = prompt.files[0] if getattr(prompt, "files", None) else None
elif st.session_state.quick_prompt:
    user_input = st.session_state.quick_prompt
    uploaded_image = None
    st.session_state.quick_prompt = ""
else:
    user_input = None
    uploaded_image = None

if user_input or uploaded_image is not None:
    if user_role == "学生端":
        if not student_name.strip() or not group_no.strip() or not topic.strip():
            st.warning("请先填写姓名、小组号和当前主题，再进行提问。")
            st.stop()

    display_user_input = user_input if user_input else "请识别这张图片中的文字。"
    api_user_input = user_input if user_input else "请识别这张图片中的文字，并结合小学图形化编程学习场景给出简要说明。"
    uploaded_image_name = ""

    # 如果上传了图片，先进行 OCR 文字识别
    if uploaded_image is not None:
        uploaded_image_name = uploaded_image.name

        with st.spinner("正在识别图片中的文字，请稍等……"):
            ocr_text = recognize_image_text(uploaded_image)

        display_user_input = f"{display_user_input}\n\n（已上传图片：{uploaded_image.name}）"

        api_user_input = f"""
{api_user_input}

【图片文字识别结果】
{ocr_text}

请结合上面的图片文字识别结果回答用户问题。

注意：
1. 如果识别结果足够，请结合识别出的文字进行分析。
2. 如果识别结果不足以判断问题，请提醒用户上传更清晰的截图，或补充说明自己想实现什么、已经做了什么、出现了什么问题。
3. 面向学生时，仍然不能直接给完整程序，只能给分步提示或检查方向。
"""

    # 页面上只显示简洁内容，不直接显示很长的 OCR 文本
    st.session_state.messages.append({
        "role": "user",
        "content": display_user_input
    })

    render_chat_bubble("user", display_user_input)

    with st.spinner("智能体正在思考中，请稍等……"):
        try:
            messages_for_api = st.session_state.messages[:-1] + [
                {
                    "role": "user",
                    "content": api_user_input
                }
            ]

            answer = call_deepseek(user_role, messages_for_api)
            render_chat_bubble("assistant", answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            log_user_input = user_input if user_input else "请识别这张图片中的文字。"
            if uploaded_image_name:
                log_user_input = f"{log_user_input}（上传图片：{uploaded_image_name}）"

            save_log(
                role=user_role,
                user_input=log_user_input,
                answer=answer,
                student_name=student_name,
                group_no=group_no,
                topic=topic
            )

        except Exception as e:
            st.error(f"调用失败：{e}")


# =========================
# 10. 侧边栏：教师端操作区
# =========================

# 只有教师端可以清空当前页面对话、下载全部记录
if user_role == "教师端":
    st.sidebar.divider()
    st.sidebar.markdown("### 对话操作")

    if st.sidebar.button("清空当前对话"):
        st.session_state.messages = []
        st.session_state.quick_prompt = ""
        st.rerun()

    logs = load_logs()

    if not logs.empty:
        st.sidebar.divider()
        st.sidebar.markdown("### 对话记录")

        csv_data = logs.to_csv(index=False, encoding="utf-8-sig")

        st.sidebar.download_button(
            label="下载全部对话记录",
            data=csv_data,
            file_name="chat_logs.csv",
            mime="text/csv"
        )
