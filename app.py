import os
import re
import base64
import html
import io
import zipfile
import json
import tempfile
from datetime import datetime


import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import time
from dotenv import load_dotenv
from openai import OpenAI
from openpyxl import Workbook
from openpyxl.drawing.image import Image as ExcelImage
from openpyxl.styles import Alignment, Font
from PIL import Image as PILImage
from streamlit_paste_button import paste_image_button
from io import BytesIO


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
    value = None

    try:
        if hasattr(st, "secrets"):
            value = st.secrets.get(key, None)
    except Exception:
        value = None

    if value is None or value == "":
        value = os.environ.get(key, default)

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
# 作品运行视频分析使用的 Qwen-VL 模型。可在 Secrets 中配置 QWEN_VL_MODEL。
QWEN_VL_MODEL = get_secret_or_env("QWEN_VL_MODEL", "qwen3-vl-flash")

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
            padding-bottom: 7.4rem;
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
            margin: 0 0 8px 38px;
            max-width: 500px;
        }

        .starter-area .stButton {
            margin-bottom: 1px;
        }

        .starter-area .stButton > button {
            width: fit-content;
            max-width: 100%;
            min-height: 22px;
            border-radius: 9px;
            border-top-left-radius: 4px;
            border: 1px solid #E5EDF6;
            background: rgba(255, 255, 255, 0.88);
            color: #44536A;
            padding: 0.12rem 0.36rem;
            text-align: left;
            font-size: 0.68rem;
            line-height: 1.2;
            box-shadow: 0 2px 6px rgba(52, 86, 130, 0.025);
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

        .chat-bubble .chat-image {
            display: block;
            max-width: 260px;
            max-height: 190px;
            border-radius: 12px;
            margin-top: 8px;
            border: 1px solid #E5EDF6;
            object-fit: contain;
            background: #FFFFFF;
        }

        .pending-image-box {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: rgba(255,255,255,0.92);
            border: 1px solid #DDE8F4;
            border-radius: 14px;
            padding: 8px 10px;
            margin: 6px 0 8px 0;
            box-shadow: 0 6px 16px rgba(52, 86, 130, 0.05);
            color: #526277;
            font-size: 0.86rem;
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

        .chat-bubble .md-table-wrap {
            width: 100%;
            overflow-x: auto;
            margin: 0.45rem 0;
        }

        .chat-bubble table.md-table {
            border-collapse: collapse;
            width: 100%;
            min-width: 760px;
            font-size: 0.88rem;
            line-height: 1.45;
            background: #FFFFFF;
        }

        .chat-bubble table.md-table th,
        .chat-bubble table.md-table td {
            border: 1px solid #DDE8F4;
            padding: 8px 9px;
            text-align: left;
            vertical-align: top;
            white-space: normal;
        }

        .chat-bubble table.md-table th {
            background: #F3F8FF;
            font-weight: 800;
            color: #21304A;
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
            bottom: 10px !important;
            background: transparent !important;
            padding-bottom: 0 !important;
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

        /* 底部截图工具条：放在输入框下方，不压住输入框，不进入侧边栏 */
        .st-key-paste_toolbar {
            position: fixed;
            left: max(350px, calc((100vw - 1240px) / 2 + 2.2rem));
            right: max(2rem, calc((100vw - 1240px) / 2 + 1rem));
            bottom: 4px !important;
            z-index: 999999;
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0 !important;
            min-height: 0 !important;
            height: auto !important;
            overflow: visible;
            pointer-events: auto;
        }

        .st-key-paste_toolbar [data-testid="stHorizontalBlock"] {
            align-items: center;
            gap: 0.28rem;
        }
        .st-key-paste_toolbar .stButton > button {
            min-height: 26px !important;
            padding: 0.12rem 0.42rem !important;
            border-radius: 11px;
            font-size: 0.78rem;
            text-align: center;
            white-space: nowrap;
            box-shadow: none;
        }

        .st-key-paste_toolbar .pending-image-box {
            margin: 0;
            padding: 3px 8px;
            font-size: 0.76rem;
            white-space: nowrap;
            box-shadow: none;
        }

        .st-key-paste_toolbar img {
            max-height: 24px !important;
            width: auto !important;
            border-radius: 7px;
            border: 1px solid #E5EDF6;
            object-fit: contain;
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
你是“图形化编程学习助手”，服务于小学高年级图形化编程课堂。

你的服务对象包括教师和学生。

总原则：
1. 教师端重在支持教学设计、课堂调控、任务单生成、问题链设计、调试提示、展示评价和教学反思。
2. 学生端重在引导思考，不能直接替学生完成完整程序，不能直接给出完整答案。
3. 所有回答必须围绕小学图形化编程任务展开。
4. 面向学生时，要用短句、分步提示，每次只推进一点。
5. 面向教师时，要规范、清晰、可操作，贴近小学课堂实际。
6.图形化编程教学流程为：情境创设、任务分析、实践创作、展示评价。
7. 任务分析建议采用：作品 → 角色 → 动作、规则、效果；必要时可引导教师和学生借助思维导图、流程图或任务分析表梳理作品结构与程序逻辑。
8. 学生调试时优先提示检查：启动事件、角色脚本归属、重复执行、条件是否满足、变量是否初始化、角色是否隐藏、造型、位置、方向是否设置正确。
9. 智能体始终遵循：教师主导，学生主体，智能辅助。
10. 不生成脱离图形化编程课堂实际的空泛内容。
11. 回答中不得出现 <br>、<p>、<div> 等 HTML 标签。
12. 学生端不使用 Markdown 表格。教师端生成教学设计时，“教学过程”必须使用 Markdown 表格呈现，表头固定为：教学环节、教师活动、学生活动、智能体支持、设计意图。
13. 教师端除教学过程表格外，其他部分可采用“小标题 + 分点说明”的形式呈现。
14. 学生端分条使用“①②③”。教师端注意编号层级：一级标题用（一）（二）（三），二级条目用1. 2. 3.，三级细目用①②③，四级细目用（1）（2）（3），不要同一层级混用。
15. 不要输出 HTML 标签，不要输出网页代码。
16. 不要输出空编号、空项目或占位符；如果信息不确定，应直接追问学生补充，而不是留下空白。
17. 不要使用 Markdown 加粗或斜体符号，例如 **重点**、*内容*、`代码`。
18. 生成教学设计时必须保证每个环节的“教师活动、学生活动、智能体支持、设计意图”完整输出；如果内容较长，应压缩前文，不得省略后文或停在“通过……”这类未完成句子。
"""


def build_role_prompt(role: str) -> str:
    if role == "教师端":
        return """
当前用户身份：教师。

你是小学高年级图形化编程教学支持助手。

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
1. 内容必须贴近小学高年级图形化编程课堂。
2. 不写空泛套话，要具体、清晰、可操作。
3. 教学流程优先采用：情境创设 → 任务分析 → 实践创作 → 展示评价。
4. 任务分析统一采用“主题 → 背景 → 角色”的方式，其中角色部分重点分析动作、规则和效果；如果作品没有背景，不要强行补充；在任务分析环节应鼓励教师使用思维导图、流程图或任务分析表帮助学生梳理作品结构、角色关系和程序逻辑。
5. 除“教学过程”外，一般不使用 Markdown 表格；“教学过程”必须使用 Markdown 表格呈现。
6. 不得出现 <br>、<p>、<div> 等 HTML 标签。
7. 教学过程表格必须包含五列：教学环节、教师活动、学生活动、智能体支持、设计意图；每个环节分别写清对应内容。
9. 生成教学设计时必须保证每个环节的“设计意图”完整输出；如果内容较长，应适当压缩前文，不要省略后文。
8. 语言要规范，适合教师直接修改后用于论文、教案或课堂材料。

二、当教师要求生成教学设计时，必须按照以下结构输出，并注意编号层级：一级标题用（一）（二）（三），二级条目用1. 2. 3.，三级细目用①②③，四级细目用（1）（2）（3），不要同一层级混用：

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

目标要结合具体图形化编程作品主题，不要空泛。

4. 教学重难点
① 教学重点：本节课学生必须掌握的核心知识、关键积木或程序结构。
② 教学难点：学生在任务分析、程序搭建或调试优化中可能遇到的主要困难。

5. 教学策略
重点说明突破重难点的策略，例如：
① 通过作品展示引出任务，激发学生创作兴趣；
② 采用“作品 → 角色 → 动作、规则、效果”的方式进行任务拆解；
③ 教师通过板书、投屏或任务单提供文字版任务分析框架，引导学生梳理作品结构、角色关系和程序逻辑；
④ 学生在思路不清时，可借助智能体整理作品逻辑、生成文字版任务分析框架或完善任务分析表，但最终分析结果应由学生自行修改和完善；
⑤ 通过教师关键示范和学生实践结合促进理解；
⑥ 通过巡视指导、同伴交流和共性问题讲解帮助学生调试修改。

6. 教学环境与资源
包括图形化编程平台、计算机或平板设备、作品素材、任务单、思维导图、流程图、评价表、智能体等。

（二）教学过程

教学过程必须使用 Markdown 表格输出，表头固定为：
| 教学环节 | 教师活动 | 学生活动 | 智能体支持 | 设计意图 |
| --- | --- | --- | --- | --- |

表格至少包含以下四个环节：情境创设、任务分析、实践创作、展示评价。每个单元格内容要具体、完整、可操作，不要只写短语。

特别注意：
1. 教学过程表格必须严格使用 Markdown 表格。
2. 每一行必须以 | 开头并以 | 结尾。
3. 每个教学环节只能占一整行，不能在单元格内部换行。
4. 单元格内不要使用 ①②③、项目符号或换行分点，可以用分号连接多个活动。
5. 不要在表格中插入空行。
6. 表格行格式示例：
| 情境创设 | 教师播放作品并提出问题，引导学生观察角色和规则 | 学生观看作品，回答角色、动作和得分方式 | 本环节不直接使用智能体，教师可课前借助智能体生成导入问题 | 激发兴趣，明确学习任务 |


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
5. 自我检查。

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

你是小学图形化编程学习助手。

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
7. 如果是任务分析，采用“作品 → 角色 → 动作、规则、效果”的方式引导；如果学生说“帮我画思维导图”“帮我梳理逻辑”“我不知道怎么分析作品”，应帮助学生整理文字版任务分析框架，但不能直接替学生完成全部内容。
8. 如果是作品优化，只给1—2个可以自己尝试的建议。
9. 要鼓励学生先尝试、运行、观察效果，再继续修改。
10. 不使用 Markdown 表格，不得出现 <br>、<p>、<div> 等 HTML 标签。
11. 分点提示统一使用“①②③”，不要使用“（1）（2）（3）”或“1. 2. 3.”。
12. 不要输出空编号、空项目或占位符；如果不知道某个角色或规则，应提醒学生补充，而不是留空。
13. 不要使用 Markdown 加粗符号，例如 **重点** 或 *内容*。
14. 不要输出单字标题，例如“被”“隐藏”“滑行”“克隆”；小标题必须是完整表达，如“被点击后怎么办”。
15. 如果一句话没有说完整，不要单独另起编号；宁可直接追问学生补充。

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

四、关于图形化编程功能解释

1. 必须区分“图形化编程内置积木可以直接实现”和“需要额外程序实现”的情况。
2. 不得把不同效果混为一谈。
3. 不得主动编造多个复杂实现方案。
4. 如果确实需要提供多个方案，必须说明哪个是基础方案，哪个是进阶方案。
5. 面向初学者时，应优先推荐课堂中最基础、最稳定的实现方式，不要一开始引入过难的坐标判断、广播、克隆等内容。

五、关于边缘处理的解释

1. “碰到边缘就反弹”是图形化编程内置积木可以直接实现的基础功能。
2. “从一端出去，从另一端出现”可以实现，但不是一个现成积木，需要用坐标判断，例如判断 x 坐标是否超过舞台边界。
3. “碰到边缘后掉头”在基础作品中通常可以用“碰到边缘就反弹”加“将旋转方式设为左右翻转”来实现。
4. 不要把“碰到边缘就反弹”和“穿屏出现”说成同一个功能。
5. 对第一节或基础任务，不主动推荐“穿屏出现”这种进阶效果。

六、关于某个主题

如果学生问“xx怎么做”，应优先引导其完成基础版。

不得一开始就把任务扩展成复杂游戏，不得主动生成过多复杂规则。

七、关于任务分析和逻辑梳理

1. 当学生说“帮我梳理逻辑”“我不知道怎么分析作品”“帮我整理思路”时，应按照“主题 → 背景 → 角色”的结构帮助学生整理。
2. 其中“角色”是重点，每个角色下面继续分析：
① 动作：这个角色要做什么；
② 规则：什么情况下触发，和谁发生关系；
③ 效果：运行后会看到什么变化。
3. 如果作品没有背景，不要写“背景”这一项。
4. 不要为了结构完整而编造不存在的角色、背景或规则。
5. 如果学生只说了作品主题，应先追问：作品里有哪些角色？每个角色要做什么？
6. 如果学生已经说明角色和功能，可以帮助学生整理成文字版任务分析框架。
7. 输出时可以采用下面这种格式：

主题：猫捉老鼠

背景：
房间背景

角色：
小猫
- 动作：根据方向键移动
- 规则：碰到老鼠时触发反馈
- 效果：老鼠被抓到或出现提示

老鼠
- 动作：随机移动或躲避小猫
- 规则：被小猫碰到后隐藏或结束游戏
- 效果：游戏出现结果反馈
"""


# =========================
# 3. 日志保存
# =========================

def save_log(
    role: str,
    user_input: str,
    answer: str,
    student_name="",
    group_no="",
    topic="",
    uploaded_image_name="",
    uploaded_image_path="",
    ocr_text="",
    uploaded_image_base64=""
):
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/chat_logs.csv"

    new_row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role,
        "student_name": student_name,
        "group_no": group_no,
        "topic": topic,
        "user_input": user_input,
        "answer": answer,
        "uploaded_image_name": uploaded_image_name,
        "uploaded_image_path": uploaded_image_path,
        "ocr_text": ocr_text,
        "uploaded_image_base64": uploaded_image_base64
    }

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        # 兼容旧版本日志：如果旧 CSV 没有新增列，自动补上空列。
        for column in new_row.keys():
            if column not in df.columns:
                df[column] = ""
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(log_path, index=False, encoding="utf-8-sig")


def load_logs():
    log_path = "logs/chat_logs.csv"
    columns = [
        "time", "role", "student_name", "group_no", "topic",
        "user_input", "answer", "uploaded_image_name", "uploaded_image_path", "ocr_text", "uploaded_image_base64"
    ]

    if not os.path.exists(log_path):
        return pd.DataFrame(columns=columns)

    if os.path.getsize(log_path) == 0:
        return pd.DataFrame(columns=columns)

    try:
        df = pd.read_csv(log_path, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(log_path, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        except Exception:
            return pd.DataFrame(columns=columns)

    for column in columns:
        if column not in df.columns:
            df[column] = ""

    return df


def save_uploaded_image_file(uploaded_file):
    """
    将学生上传的图片保存到 logs/uploads 文件夹。
    CSV 中会记录图片文件名和相对路径；教师下载 ZIP 时会一起下载原图。
    """
    if uploaded_file is None:
        return "", ""

    upload_dir = os.path.join("logs", "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    original_name = os.path.basename(uploaded_file.name or "uploaded_image.png")
    safe_name = re.sub(r"[^0-9a-zA-Z_\-.\u4e00-\u9fff]", "_", original_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved_name = f"{timestamp}_{safe_name}"
    saved_path = os.path.join(upload_dir, saved_name)

    uploaded_file.seek(0)
    with open(saved_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    uploaded_file.seek(0)

    return saved_name, saved_path


def build_logs_zip_bytes():
    """
    打包下载全部对话记录和上传图片。
    ZIP 中包含：
    1. chat_logs.csv
    2. uploads/ 文件夹中的原始截图
    """
    buffer = io.BytesIO()
    log_path = os.path.join("logs", "chat_logs.csv")
    upload_dir = os.path.join("logs", "uploads")

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        if os.path.exists(log_path):
            zip_file.write(log_path, arcname="chat_logs.csv")

        if os.path.exists(upload_dir):
            for root, _, files in os.walk(upload_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    arcname = os.path.relpath(file_path, "logs")
                    zip_file.write(file_path, arcname=arcname)

    buffer.seek(0)
    return buffer.getvalue()


def create_excel_with_images(logs_df):
    """
    生成包含全部对话记录的 Excel。
    每一行对应一次用户提问和智能体回答；
    上传图片会嵌入到“上传截图”列，教师打开 Excel 可以直接看到图片。
    新记录会同时保存图片路径和 base64；如果路径失效，会尝试从 base64 恢复图片。
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "对话记录"

    headers = [
        "时间", "身份", "小组号", "当前主题",
        "学生/教师问题", "上传截图", "智能体回答", "OCR识别文字"
    ]
    ws.append(headers)

    column_widths = {
        "A": 20,
        "B": 12,
        "C": 12,
        "D": 16,
        "E": 42,
        "F": 36,
        "G": 70,
        "H": 48,
    }
    for col, width in column_widths.items():
        ws.column_dimensions[col].width = width

    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")

    def safe_cell(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value)

    def clean_question_for_excel(value):
        """Excel 的问题列中不再重复显示“上传图片：xxx”，图片由截图列直接展示。"""
        text = safe_cell(value)
        text = re.sub(r"\n*（已上传图片：.*?）", "", text)
        text = re.sub(r"（上传图片：.*?）", "", text)
        return text.strip()

    def resolve_image_path(raw_path):
        """兼容相对路径、旧路径、Windows/Unix 分隔符。"""
        raw_path = safe_cell(raw_path).strip()
        if not raw_path or raw_path.lower() == "nan":
            return ""

        normalized = raw_path.replace("\\", os.sep).replace("/", os.sep)
        candidates = [
            raw_path,
            normalized,
            os.path.abspath(raw_path),
            os.path.abspath(normalized),
            os.path.join(os.getcwd(), raw_path),
            os.path.join(os.getcwd(), normalized),
        ]

        # 如果 CSV 里只保存了文件名，也尝试到 logs/uploads 里找
        basename = os.path.basename(normalized)
        if basename:
            candidates.append(os.path.join("logs", "uploads", basename))
            candidates.append(os.path.join(os.getcwd(), "logs", "uploads", basename))

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            if os.path.exists(candidate):
                return candidate
        return ""

    def normalize_image_to_temp_file(image_path="", image_b64="", row_no=0):
        """
        将图片统一转成临时 PNG 文件，再交给 openpyxl 插入。
        这样比直接把 BytesIO 交给 ExcelImage 更稳定。
        """
        try:
            if image_path and os.path.exists(image_path):
                pil_img = PILImage.open(image_path)
            elif image_b64:
                raw = base64.b64decode(image_b64)
                pil_img = PILImage.open(io.BytesIO(raw))
            else:
                return ""

            pil_img = pil_img.convert("RGB")
            pil_img.thumbnail((260, 180))

            temp_dir = os.path.join("logs", "excel_temp_images")
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"excel_row_{row_no}.png")
            pil_img.save(temp_path, format="PNG")
            return temp_path
        except Exception:
            return ""

    for idx, row in logs_df.iterrows():
        excel_row = idx + 2

        ws.cell(excel_row, 1, safe_cell(row.get("time", "")))
        ws.cell(excel_row, 2, safe_cell(row.get("role", "")))
        ws.cell(excel_row, 3, safe_cell(row.get("group_no", "")))
        ws.cell(excel_row, 4, safe_cell(row.get("topic", "")))
        ws.cell(excel_row, 5, clean_question_for_excel(row.get("user_input", "")))
        ws.cell(excel_row, 7, safe_cell(row.get("answer", "")))
        ws.cell(excel_row, 8, safe_cell(row.get("ocr_text", "")))

        # 行高要足够，否则图片虽然插入了，但看起来像“没显示”。
        ws.row_dimensions[excel_row].height = 145

        image_path = resolve_image_path(row.get("uploaded_image_path", ""))
        image_b64 = safe_cell(row.get("uploaded_image_base64", ""))
        image_name = safe_cell(row.get("uploaded_image_name", ""))

        temp_image_path = normalize_image_to_temp_file(image_path, image_b64, excel_row)
        if temp_image_path:
            try:
                excel_img = ExcelImage(temp_image_path)
                excel_img.width = 240
                excel_img.height = 160
                ws.add_image(excel_img, f"F{excel_row}")
            except Exception:
                ws.cell(excel_row, 6, f"图片插入失败：{image_name or image_path}")
        elif image_name:
            ws.cell(excel_row, 6, f"图片文件未找到：{image_name}")
        else:
            ws.cell(excel_row, 6, "")

        for col_idx in range(1, 9):
            ws.cell(excel_row, col_idx).alignment = Alignment(wrap_text=True, vertical="top")

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output.getvalue()


# =========================
# 4. 清理模型回复
# =========================

def clean_answer(text: str) -> str:
    """
    清理模型回复：
    1. 删除不希望出现的 HTML 标签；
    2. 删除流式输出残留符号；
    3. 删除空编号、空项目；
    4. 清理多余空行。
    注意：这里不再删除 Markdown 表格竖线，也不强制把所有编号转成①②③。
    学生端编号统一交给 normalize_student_numbering()；
    教师端保留（一）/1./①等层级编号，并允许教学过程表格正常显示。
    """
    if not text:
        return ""

    for bad_char in ["▌", "█", "▐", "▍", "▎", "▏", "■", "□", "▪", "▫"]:
        text = text.replace(bad_char, "")

    text = text.replace("**", "")
    text = text.replace("*", "")
    text = text.replace("`", "")

    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</div\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<div\s*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?span[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?strong[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?b[^>]*>", "", text, flags=re.IGNORECASE)
    text = text.replace("&nbsp;", " ")

    text = re.sub(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[（(]?\d{1,2}[）).、．]\s*$", "", text, flags=re.MULTILINE)

    incomplete_words = r"被|隐藏|显示|滑行|克隆|碰到|点击|分数|变量|造型|位置|条件"
    text = re.sub(rf'^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*[‘’\'"“”]?({incomplete_words})[‘’\'"“”]?\s*$', "", text, flags=re.MULTILINE)
    text = re.sub(rf'^\s*[（(]?\d{{1,2}}[）).、．]\s*[‘’\'"“”]?({incomplete_words})[‘’\'"“”]?\s*$', "", text, flags=re.MULTILINE)

    text = re.sub(r"^\s*[:：]\s*$", "", text, flags=re.MULTILINE)

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def normalize_student_numbering(text: str) -> str:
    """
    只用于学生端：把常见的 1.、（1）、(1)、1、 等编号统一成 ①②③。
    教师端不要调用这个函数，避免破坏正式层级编号。
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
            for mark in [".", "．", "、", ")"]:
                marker = f"{num}{mark}"
                if stripped.startswith(marker):
                    new_line = prefix + circled + " " + stripped[len(marker):].lstrip()
                    break

            if new_line != line:
                break

        converted_lines.append(new_line)

    text = "\n".join(converted_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def postprocess_answer(role: str, text: str) -> str:
    """
    统一清理模型回答。
    学生端额外统一编号；教师端保留层级编号。
    """
    answer = clean_answer(text)

    if role == "学生端":
        answer = normalize_student_numbering(answer)

    return answer

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
    支持普通分行、标题、项目符号、编号行，以及教师端教学过程常用的 Markdown 表格。
    """
    content = content or ""

    circled_numbers = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧", "⑨", "⑩"]
    parts = []
    lines = content.splitlines()
    i = 0

    def is_table_line(line: str) -> bool:
        s = line.strip()
        return s.startswith("|") and s.endswith("|") and s.count("|") >= 2

    def is_separator_line(line: str) -> bool:
        s = line.strip().strip("|").strip()
        if not s:
            return False
        cells = [c.strip() for c in s.split("|")]
        return all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)

    def split_table_cells(line: str):
        return [html.escape(c.strip()) for c in line.strip().strip("|").split("|")]

    def render_table(table_lines):
        if len(table_lines) < 2:
            return ""
        header = split_table_cells(table_lines[0])
        body_start = 1
        if len(table_lines) > 1 and is_separator_line(table_lines[1]):
            body_start = 2
        rows = [split_table_cells(row) for row in table_lines[body_start:] if is_table_line(row)]
        html_parts = ['<div class="md-table-wrap"><table class="md-table">']
        html_parts.append("<thead><tr>")
        for cell in header:
            html_parts.append(f"<th>{cell}</th>")
        html_parts.append("</tr></thead>")
        if rows:
            html_parts.append("<tbody>")
            for row in rows:
                html_parts.append("<tr>")
                for idx in range(len(header)):
                    cell = row[idx] if idx < len(row) else ""
                    html_parts.append(f"<td>{cell}</td>")
                html_parts.append("</tr>")
            html_parts.append("</tbody>")
        html_parts.append("</table></div>")
        return "".join(html_parts)

    while i < len(lines):
        line = lines[i]
        stripped_raw = line.strip()

        if is_table_line(line):
            table_lines = []
            while i < len(lines) and (is_table_line(lines[i]) or is_separator_line(lines[i])):
                table_lines.append(lines[i])
                i += 1
            table_html = render_table(table_lines)
            if table_html:
                parts.append(table_html)
            continue

        stripped = html.escape(stripped_raw)
        stripped = apply_simple_bold(stripped)

        if not stripped_raw:
            parts.append('<div class="md-space"></div>')
            i += 1
            continue

        if stripped_raw.startswith("#"):
            heading_text = html.escape(stripped_raw.lstrip("#").strip())
            if heading_text:
                parts.append('<div class="md-heading">' + heading_text + '</div>')
                i += 1
                continue

        if stripped_raw.startswith("- "):
            parts.append('<div class="md-bullet">• ' + html.escape(stripped_raw[2:].strip()) + '</div>')
            i += 1
            continue

        if stripped_raw.startswith("• "):
            parts.append('<div class="md-bullet">' + html.escape(stripped_raw) + '</div>')
            i += 1
            continue

        is_number_line = False
        for number_text in circled_numbers:
            if stripped_raw.startswith(number_text):
                is_number_line = True
                break

        for number in range(1, 21):
            if stripped_raw.startswith(str(number) + ".") or stripped_raw.startswith(str(number) + "．") or stripped_raw.startswith(str(number) + "、"):
                is_number_line = True
                break

        if re.match(r"^[（(]\d{1,2}[）)]", stripped_raw):
            is_number_line = True

        if is_number_line:
            parts.append('<div class="md-number">' + stripped + '</div>')
        else:
            parts.append('<div class="md-line">' + stripped + '</div>')

        i += 1

    return "".join(parts)


def build_chat_bubble_html(role: str, content: str, image_base64: str = "") -> str:
    """
    构建聊天气泡 HTML。
    如果用户消息带有截图，则把截图嵌入到同一个气泡中，避免“上一张图片消失”。
    助手消息在最终渲染前再统一一次编号，避免出现 ①②③（4）⑤ 混用。
    """
    display_content = content or ""
    safe_content = format_message_content(display_content)

    image_html = ""
    if image_base64:
        image_html = f"""
        <img class="chat-image" src="data:image/png;base64,{image_base64}" />
        """

    if role == "user":
        bubble_role = "user"
        avatar = "我"
    else:
        bubble_role = "assistant"
        avatar = "🐱"

    return f"""
    <div class="chat-wrap {bubble_role}">
        <div class="chat-avatar {bubble_role}">{avatar}</div>
        <div class="chat-content">
            <div class="chat-bubble {bubble_role}">{safe_content}{image_html}</div>
        </div>
    </div>
    """


def render_chat_bubble(role: str, content: str, image_base64: str = ""):
    """
    使用自定义 HTML 渲染类似微信的左右聊天气泡。
    """
    st.markdown(
        build_chat_bubble_html(role, content, image_base64),
        unsafe_allow_html=True
    )

def scroll_to_bottom(holder=None, smooth: bool = True):
    """
    尽量让页面自动滚动到底部。
    这个版本只滚动页面和主容器，不再强制 chat_input 进入视野，
    避免右侧滚动条反复向上/向下跳动。
    """
    behavior = "smooth" if smooth else "auto"
    stamp = str(time.time()).replace(".", "")

    script = f"""
    <script id="scroll-bottom-{stamp}">
    function scrollBottom{stamp}() {{
        try {{
            const doc = window.parent.document;
            const targets = [];

            if (doc.scrollingElement) targets.push(doc.scrollingElement);
            if (doc.documentElement) targets.push(doc.documentElement);
            if (doc.body) targets.push(doc.body);

            const selectors = [
                '[data-testid="stAppViewContainer"]',
                '[data-testid="stMain"]',
                'section.main',
                '.main'
            ];

            selectors.forEach(function(sel) {{
                const el = doc.querySelector(sel);
                if (el) targets.push(el);
            }});

            targets.forEach(function(el) {{
                try {{
                    const maxTop = Math.max(el.scrollHeight, el.offsetHeight, el.clientHeight) + 2000;
                    if (typeof el.scrollTo === "function") {{
                        el.scrollTo({{ top: maxTop, behavior: "{behavior}" }});
                    }}
                    el.scrollTop = maxTop;
                }} catch (e) {{}}
            }});
        }} catch (e) {{}}
    }}

    setTimeout(scrollBottom{stamp}, 60);
    setTimeout(scrollBottom{stamp}, 180);
    setTimeout(scrollBottom{stamp}, 420);
    setTimeout(scrollBottom{stamp}, 800);
    </script>
    """

    if holder is not None:
        with holder.container():
            components.html(script, height=0)
    else:
        components.html(script, height=0)


# =========================
# 4.5 图片文字识别 OCR
# =========================
def analyze_sb3_file(sb3_file):
    """
    解析图形化编程.sb3 文件，提取用于作品评价和与教师基础作品对比的结构信息。
    """
    result = {
        "file_name": sb3_file.name,
        "sprites": [],
        "stage_count": 0,
        "sprite_count": 0,
        "block_count": 0,
        "event_blocks": 0,
        "loop_blocks": 0,
        "condition_blocks": 0,
        "variable_blocks": 0,
        "broadcast_blocks": 0,
        "looks_blocks": 0,
        "motion_blocks": 0,
        "sound_blocks": 0,
        "operators_blocks": 0,
        "sensing_blocks": 0,
        "opcodes": [],
        "summary": ""
    }

    try:
        sb3_file.seek(0)
        with zipfile.ZipFile(sb3_file, "r") as z:
            with z.open("project.json") as f:
                project = json.load(f)

        targets = project.get("targets", [])

        for target in targets:
            name = target.get("name", "")
            is_stage = target.get("isStage", False)
            blocks = target.get("blocks", {})

            if is_stage:
                result["stage_count"] += 1
            else:
                result["sprite_count"] += 1
                result["sprites"].append(name)

            for block_id, block in blocks.items():
                if not isinstance(block, dict):
                    continue

                opcode = block.get("opcode", "")
                if not opcode:
                    continue

                result["block_count"] += 1
                result["opcodes"].append(opcode)

                if opcode.startswith("event_"):
                    result["event_blocks"] += 1
                elif opcode.startswith("control_repeat") or opcode.startswith("control_forever"):
                    result["loop_blocks"] += 1
                elif opcode.startswith("control_if"):
                    result["condition_blocks"] += 1
                elif opcode.startswith("data_"):
                    result["variable_blocks"] += 1
                elif "broadcast" in opcode:
                    result["broadcast_blocks"] += 1
                elif opcode.startswith("looks_"):
                    result["looks_blocks"] += 1
                elif opcode.startswith("motion_"):
                    result["motion_blocks"] += 1
                elif opcode.startswith("sound_"):
                    result["sound_blocks"] += 1
                elif opcode.startswith("operator_"):
                    result["operators_blocks"] += 1
                elif opcode.startswith("sensing_"):
                    result["sensing_blocks"] += 1

        result["opcodes"] = sorted(list(set(result["opcodes"])))

        result["summary"] = (
            f"作品包含 {result['sprite_count']} 个角色，"
            f"共检测到 {result['block_count']} 个积木；"
            f"事件积木 {result['event_blocks']} 个，"
            f"循环积木 {result['loop_blocks']} 个，"
            f"条件积木 {result['condition_blocks']} 个，"
            f"变量相关积木 {result['variable_blocks']} 个，"
            f"广播相关积木 {result['broadcast_blocks']} 个。"
        )

    except Exception as e:
        result["summary"] = f"解析失败：{e}"

    return result


def extract_eval_field(text, field_name):
    """
    从模型输出中提取指定字段。
    支持字段前带编号，也支持多行字段内容。
    """
    if not text:
        return ""

    field_names = [
        "核心功能完成情况",
        "完成等级",
        "各维度评分及依据",
        "建议总分",
        "运行画面分析",
        "总评依据",
        "评分依据",
        "改进建议",
        "教师复核点",
        # 兼容旧版本输出
        "维度评分",
        "完整性建议分",
        "技术性建议分",
        "创新性建议分",
        "艺术性建议分",
    ]

    field_pattern = r"(?:[（(]?[一二三四五六七八九十\d]+[）)]?[、.．]?\s*)?"
    all_fields = "|".join(map(re.escape, field_names))

    pattern = (
        field_pattern
        + re.escape(field_name)
        + r"\s*[:：]\s*"
        + r"(.*?)"
        + r"(?=\n\s*"
        + field_pattern
        + r"(?:"
        + all_fields
        + r")\s*[:：]|\Z)"
    )

    match = re.search(pattern, text, flags=re.S)

    if not match:
        return ""

    value = match.group(1).strip()
    value = re.sub(r"\n{2,}", "\n", value)
    return value


def extract_score_number(text):
    """
    从“25分”“建议25”“25/30”等文本中提取第一个数字。
    用于写入 Excel 的总分字段。
    """
    if not text:
        return ""

    match = re.search(r"\d+(?:\.\d+)?", str(text))
    if not match:
        return ""

    number = match.group(0)

    if "." in number:
        return float(number)

    return int(number)


def normalize_completion_level(level_text):
    """
    完成等级只保留等级名称，不保留括号、分数区间或解释。
    """
    text = str(level_text or "").strip()
    for level in ["优秀完成", "良好完成", "基本完成", "部分完成", "未完成"]:
        if level in text:
            return level
    return text


def parse_dimension_evaluations(eval_text):
    """
    解析“各维度评分及依据”字段。
    期望模型输出格式：
    各维度评分及依据：
    完整性评分及依据：26/30。依据：……
    技术性评分及依据：25/30。依据：……

    返回：
    {
        "完整性评分及依据": "26/30。依据：……",
        "技术性评分及依据": "25/30。依据：……"
    }
    """
    result = {}

    block = extract_eval_field(eval_text, "各维度评分及依据")
    if not block:
        # 兼容旧版本“维度评分”字段
        block = extract_eval_field(eval_text, "维度评分")

    if not block:
        return result

    lines = [line.strip() for line in block.splitlines() if line.strip()]

    current_key = ""
    current_value_parts = []

    def flush_current():
        if current_key:
            value = " ".join(part.strip() for part in current_value_parts if part.strip()).strip()
            if value:
                result[current_key] = value

    for line in lines:
        # 支持：完整性评分及依据：26/30。依据：……
        m = re.match(r"^(.{1,30}?评分及依据)\s*[:：]\s*(.+)$", line)
        if m:
            flush_current()
            current_key = m.group(1).strip()
            current_value_parts = [m.group(2).strip()]
            continue

        # 支持：完整性：26/30。依据：……
        m = re.match(r"^(.{1,20}?)\s*[:：]\s*(\d+(?:\.\d+)?\s*/\s*\d+.*)$", line)
        if m:
            flush_current()
            dim_name = m.group(1).strip()
            if not dim_name.endswith("评分及依据"):
                dim_name = f"{dim_name}评分及依据"
            current_key = dim_name
            current_value_parts = [m.group(2).strip()]
            continue

        # 支持多行依据接在上一行后面
        if current_key:
            current_value_parts.append(line)

    flush_current()
    return result


TEACHER_CASE_RULES = """
【教师评价案例集形成的评分尺度】

本评分尺度来自研究者与信息科技教师共同评价学生作品后形成的评价案例集。
该规则只用于帮助智能体判断完成等级和评分尺度，不用于决定评价维度。

一、完成等级与总分区间
1. 优秀完成：90—95分
核心功能全部完成，程序运行稳定，没有明显错误，并有一定优化、拓展或创意表现。

2. 良好完成：85—89分
核心功能基本完整，程序能够正常运行，但存在较小缺陷、程序结构问题或表现不够完善。

3. 基本完成：75—84分
完成了大部分核心功能，但缺失一项关键功能，或部分交互、反馈、运行效果不完整。

4. 部分完成：60—74分
只完成了部分核心功能，作品能够呈现一定效果，但尚未形成完整作品逻辑。

5. 未完成：35—59分
核心功能未实现，程序无法形成有效作品，或主要角色、交互、运行逻辑明显缺失。

二、评分原则
1. 评价时必须优先判断本课核心功能完成情况。
2. 教师上传的评价量表决定评价维度和各维度分值。
3. 教师评价案例集只用于确定评分尺度，不得替代教师上传的评价量表。
4. 如果课堂任务或评价量表没有明确要求变量、广播、倒计时、胜负界面、关卡切换，不得因为缺失这些功能直接扣分。
5. 如果学生作品实现了角色控制、移动、碰撞检测、消息反馈、造型变化、重新出现等核心交互，应给予合理肯定。
6. 如果作品存在明显运行错误，例如角色方向错误、无法控制、核心角色不动、碰撞后无反馈、程序只能运行一次，应根据影响程度降低完成等级。
7. 如果视频关键帧没有完整呈现某项功能，不能直接判定该功能缺失，应写入“教师复核点”。
8. 如果学生作品已经实现本课核心功能，程序能够运行，且作品能够表达任务目标，一般应评为良好完成或优秀完成，不应仅因缺少扩展功能而降为基本完成。
9. 如果学生作品虽然能看到主要效果，但存在“只能运行一次、抓到后不能重新开始、角色控制方式明显不符合任务、关键反馈缺失、运行逻辑不稳定”等问题，应降为基本完成或部分完成。
"""


def load_teacher_cases_text():
    """
    读取后端内置的教师评价案例集。
    文件名固定为 teacher_cases.xlsx，放在和本 .py 文件同一目录。
    案例集用于帮助智能体学习教师评分尺度，不要求普通教师每次上传。
    """
    case_path = "teacher_cases.xlsx"

    if not os.path.exists(case_path):
        return "未检测到 teacher_cases.xlsx，本次仅依据评分规则和教师上传评价量表进行评价。"

    try:
        df = pd.read_excel(case_path)
    except Exception as e:
        return f"teacher_cases.xlsx 读取失败：{e}"

    needed_cols = ["作品编号", "主题","完成等级", "建议总分", "评分依据", "改进建议"]
    for col in needed_cols:
        if col not in df.columns:
            df[col] = ""

    case_lines = []
    for _, row in df.iterrows():
        case_text = (
            f"案例{row.get('作品编号', '')}\n"
            f"主题：{row.get('主题', '')}\n"
            f"完成等级：{row.get('完成等级', '')}\n"
            f"教师评分：{row.get('建议总分', '')}\n"
            f"评分依据：{row.get('评分依据', '')}\n"
            f"改进建议：{row.get('改进建议', '')}\n"
        )
        case_lines.append(case_text)

    return "\n---\n".join(case_lines[:40])


def evaluate_project_with_rubric(sb3_analysis, rubric_text, reference_analysis=None, video_analysis=None, reference_video_analysis=None):
    """
    根据评价量表、教师基础作品、教师基础版运行视频、学生作品结构和学生运行视频生成辅助评价。
    教师基础版运行视频用于提供当前任务的“可见运行效果参照”，不再要求教师填写文字版运行效果说明。
    """
    reference_text = ""
    teacher_cases_text = load_teacher_cases_text()

    if reference_analysis or reference_video_analysis:
        reference_text = f"""
【教师基础作品参照】
教师基础作品和教师基础版运行视频只作为“核心功能目标参照”，不是标准答案代码。
学生不需要和教师作品使用完全相同的积木或实现方式，只要实现相同或相近的核心运行效果即可。

【教师基础作品结构】
{json.dumps(reference_analysis, ensure_ascii=False, indent=2) if reference_analysis else "未上传教师基础作品 .sb3。"}

【教师基础版运行视频分析】
以下内容来自 Qwen-VL 对教师基础版作品运行视频关键帧的观察，用于帮助判断本节课作品应呈现的核心运行效果。
{json.dumps(reference_video_analysis, ensure_ascii=False, indent=2) if reference_video_analysis else "未上传教师基础版运行视频。"}
"""

    if video_analysis:
        video_text = f"""
【学生作品运行视频分析】
以下内容来自 Qwen-VL 对学生作品运行录屏关键帧的观察，只作为运行效果参考。
如果视频分析与 .sb3 结构分析不一致，请在“教师复核点”中提示教师进一步核对。
{json.dumps(video_analysis, ensure_ascii=False, indent=2)}
"""
    else:
        video_text = """
【学生作品运行视频分析】
未上传学生作品运行视频，本次评价主要依据学生作品结构分析、教师基础作品、教师基础版运行视频和评价量表。
"""

    prompt = f"""
你是小学五年级图形化编程作品辅助评价助手。

请根据教师上传的作品评价量表、学生作品结构分析结果、学生作品运行视频分析、教师基础作品结构、教师基础版运行视频分析，以及教师评价案例集形成的评分尺度，生成作品辅助评价建议。

【最重要的评价规则】
1. 必须严格依据教师上传的评价量表进行评价。
2. 评价维度必须来自教师上传的评价量表，不得固定使用“完整性、技术性、创新性、艺术性”等系统预设维度。
3. 每个评价维度都必须给出“建议分数”和“评分依据”。
4. 每个维度的评分依据必须对应该维度在教师上传评价量表中的评价要求。
5. 如果教师上传的评价量表包含“任务完成度、编程思维、作品表达”等维度，就必须按这些维度评分。
6. 如果教师上传的评价量表包含“完整性、技术性、创新性、艺术性”等维度，才可以按这些维度评分。
7. 教师评价案例集只用于学习评分尺度和完成等级。
8. 教师基础作品和教师基础版运行视频共同作为核心功能目标参照，不是标准答案代码。
9. 教师基础版运行视频用于帮助理解本节课基础作品应呈现的可见效果；不得把教师基础版作品中的所有细节都当作学生必须完全一致完成的标准。
10. 必须给出具体分数，不要只给区间。
11. 评价时应先结合教师基础版作品结构、教师基础版运行视频、学生作品结构和学生运行视频，判断学生作品是否实现主要运行效果，再确定完成等级和建议总分。
12. 评分时应采用小学五年级课堂作品评价标准。只要学生作品已经实现本课主要交互和运行目标，即使程序结构不够规范、画面表现一般或缺少拓展功能，也不应过度压低分数。
13. 如果学生作品已经实现教师基础版作品中的主要运行效果，程序能够正常运行，交互基本有效，一般可评为良好完成；如果在此基础上还有较好的拓展或表现效果，可评为优秀完成。
14. 不得将教师基础作品中的变量、广播、倒计时、胜负界面、音效、关卡等具体实现细节自动视为学生必须完成的项目，除非教师上传的评价量表明确要求。
15. 如果学生作品只实现了部分主要运行效果，或运行过程中存在较明显问题，但仍能看出作品基本目标，通常评为基本完成。
16. 如果学生作品只能呈现少量效果，主要交互不完整，或运行逻辑明显不稳定，通常评为部分完成。
17. 如果学生作品无法形成有效作品，主要角色、交互或运行逻辑明显缺失，通常评为未完成。
18. 视频关键帧只能作为运行效果参考；如果视频没有观察到某项功能，但作品结构中可能存在相关程序，不能直接判定为缺失，应写入教师复核点。
19. 不得因为视频未完整呈现某项功能，就直接大幅降低分数；也不得因为画面角色较多或背景完整，就忽略核心交互缺失。
20. 建议总分必须等于各维度建议分数之和。
21. 如果各维度分数相加后的总分与完成等级区间不一致，应调整各维度分数，使建议总分落入对应完成等级区间，并保持评分依据一致。

【完成等级要求】
必须先判断完成等级，再根据完成等级确定建议总分。

建议总分必须落在该完成等级对应的分数区间内：
优秀完成：90—95分
良好完成：85—89分
基本完成：75—84分
部分完成：60—74分
未完成：35—59分

完成等级只能填写以下五个之一：
优秀完成
良好完成
基本完成
部分完成
未完成

不得在完成等级后添加分数区间或解释。

【教师评价案例评分尺度】
{TEACHER_CASE_RULES}

【教师评价案例集】
{teacher_cases_text}

请参考教师评价案例集中评分依据，理解教师评分尺度。
当学生作品与案例中的完成程度相近时，评分应保持一致。
如果案例集与通用评分规则存在差异，应优先参考案例集中体现的评分宽严尺度，但不得改变教师上传评价量表中的评价维度、满分设置和核心评价要求。
案例集只用于学习教师评分尺度，不用于替代教师上传的评价量表。

【输出格式要求】
请严格按照以下格式输出，不要添加额外标题。
字段名前不要添加（一）（二）（三）或1. 2. 3.。
不要输出“维度评分”字段。
各维度评分及依据必须严格使用教师评价量表中的真实维度名称。
核心功能完成情况：
完成等级：
各维度评分及依据：
请逐行输出教师量表中的真实维度名称，例如：
任务完成度评分及依据：36/40。依据：……
编程思维评分及依据：25/30。依据：……
建议总分：
运行画面分析：
总评依据：
改进建议：
教师复核点：

示例1：如果教师量表是“任务完成度40分、编程思维30分、作品表达30分”，则“各维度评分及依据”写成：
任务完成度评分及依据：36/40。依据：……
编程思维评分及依据：25/30。依据：……
作品表达评分及依据：26/30。依据：……

示例2：如果教师量表是“完整性30分、技术性30分、创新性20分、艺术性20分”，则“各维度评分及依据”写成：
完整性评分及依据：28/30。依据：……
技术性评分及依据：26/30。依据：……
创新性评分及依据：17/20。依据：……
艺术性评分及依据：18/20。依据：……

【学生作品结构分析】
{json.dumps(sb3_analysis, ensure_ascii=False, indent=2)}

{reference_text}

{video_text}

【教师上传的作品评价量表】
{rubric_text}
"""

    messages = [
        {"role": "user", "content": prompt}
    ]

    return call_deepseek_full("教师端", messages)

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

    try:
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

        raw_text = response.choices[0].message.content or ""
        return clean_answer(raw_text)

    except Exception as e:
        return f"图片文字识别失败：{e}"


# =========================
# 4.6 Qwen-VL：学生作品运行视频分析
# =========================

def save_uploaded_video_to_temp(video_file):
    """
    将 Streamlit 上传的视频临时保存到本地，供 OpenCV 抽取关键帧。
    """
    suffix = os.path.splitext(video_file.name or "uploaded_video.mp4")[1] or ".mp4"
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    video_file.seek(0)
    temp.write(video_file.getvalue())
    temp.flush()
    temp.close()
    video_file.seek(0)
    return temp.name


def pil_image_to_data_url(pil_img, image_format="JPEG"):
    """
    将 PIL 图片转为 Qwen-VL 可读取的 data URL。
    """
    buffer = io.BytesIO()
    pil_img = pil_img.convert("RGB")
    pil_img.thumbnail((960, 540))
    pil_img.save(buffer, format=image_format, quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if image_format.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{encoded}"


def decide_video_frame_count(duration):
    """
    根据视频时长自动决定抽取几帧。
    """
    if duration <= 15:
        return 5
    elif duration <= 30:
        return 6
    elif duration <= 60:
        return 8
    else:
        return 10


def extract_video_keyframes(video_file):
    """
    自动根据视频时长均匀抽取关键帧。
    短视频也能抽到开始、中间、结尾，避免12秒视频只抽2-3帧。
    返回：[{time, image, data_url}, ...]
    """
    try:
        import cv2
    except Exception:
        st.warning("当前环境没有安装 opencv-python-headless，无法抽取视频关键帧。")
        return []

    temp_video_path = save_uploaded_video_to_temp(video_file)
    frames = []

    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total_frames / fps if fps else 0

        if duration <= 0:
            candidate_times = [0]
        else:
            frame_count = decide_video_frame_count(duration)

            if frame_count <= 1:
                candidate_times = [0]
            else:
                candidate_times = [
                    round(i * duration / (frame_count - 1), 2)
                    for i in range(frame_count)
                ]

        for second in candidate_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
            success, frame = cap.read()

            if not success or frame is None:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(frame_rgb)

            frames.append({
                "time": second,
                "image": pil_img,
                "data_url": pil_image_to_data_url(pil_img)
            })

        cap.release()

    finally:
        try:
            os.remove(temp_video_path)
        except Exception:
            pass

    return frames


def analyze_video_frame_with_qwen_vl(frame_item, task_context=""):
    """
    使用 Qwen-VL 分析单张关键帧中的图形化编程运行效果。
    """
    if ocr_client is None:
        return "未配置 DASHSCOPE_API_KEY，无法调用 Qwen-VL 分析视频关键帧。"

    prompt = f"""
你是小学图形化编程作品运行效果观察助手。
请只根据这一帧画面，判断学生作品运行时呈现出的可见现象。

观察重点：
1. 是否能看到图形化编程舞台或作品运行画面；
2. 角色是否存在、位置是否正常；
3. 是否能看到分数、倒计时、提示文字、胜利/失败等信息；
4. 是否能看出角色移动、碰撞、场景切换、交互反馈等运行效果；
5. 是否存在明显异常，例如画面空白、角色消失、卡住、变量显示异常。

注意：
1. 不要凭空猜测代码。
2. 如果单帧无法判断动态过程，请明确写“单帧无法确认”。
3. 语言简洁，适合放入教师评价表。

课堂任务或评价量表背景：
{task_context[:1000]}

当前关键帧时间：第 {frame_item.get('time', 0)} 秒
"""
    try:
        response = ocr_client.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": frame_item["data_url"]}}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=700
        )

        raw_text = response.choices[0].message.content or ""
        return clean_answer(raw_text)

    except Exception as e:
        return f"该关键帧调用 Qwen-VL 分析失败：{e}"


def summarize_video_analysis_with_qwen_vl(frame_observations, task_context=""):
    """
    汇总多个关键帧观察结果，形成作品运行视频分析。
    """
    if not frame_observations:
        return "未获得有效视频关键帧，无法进行运行画面分析。"

    if ocr_client is None:
        return "未配置 DASHSCOPE_API_KEY，无法调用 Qwen-VL 汇总视频分析。"

    observations_text = "\n".join([
        f"第{item['time']}秒：{item['observation']}" for item in frame_observations
    ])

    prompt = f"""
你是小学五年级图形化编程作品运行视频辅助评价助手。
请根据多个关键帧观察结果，汇总学生作品的运行表现。

要求：
1. 重点判断作品是否有可见运行效果；
2. 说明角色、变量、提示文字、场景变化、交互反馈等是否可见；
3. 区分“已经观察到”和“关键帧无法确认”；
4. 不替代教师正式评分，只作为教师复核参考；
5. 输出简洁，便于写入 Excel。

请严格按以下格式输出：
运行画面分析：
可确认的运行效果：
无法确认或需教师复核：

课堂任务或评价量表背景：
{task_context[:1200]}

关键帧观察结果：
{observations_text}
"""

    try:
        response = ocr_client.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1200
        )

        raw_text = response.choices[0].message.content or ""
        return clean_answer(raw_text)

    except Exception as e:
        return f"视频关键帧汇总分析失败：{e}"


def analyze_project_video_with_qwen_vl(video_file, task_context=""):
    """
    完整视频分析流程：上传视频 → 抽取关键帧 → Qwen-VL 分析单帧 → 汇总运行表现。
    """
    frames = extract_video_keyframes(video_file)

    if not frames:
        return {
            "video_file": video_file.name if video_file else "",
            "frame_count": 0,
            "frame_observations": [],
            "summary": "未能抽取有效关键帧，无法分析运行视频。"
        }

    frame_observations = []
    for frame_item in frames:
        observation = analyze_video_frame_with_qwen_vl(frame_item, task_context=task_context)
        frame_observations.append({
            "time": frame_item.get("time", 0),
            "observation": observation
        })

    summary = summarize_video_analysis_with_qwen_vl(
        frame_observations,
        task_context=task_context
    )

    return {
        "video_file": video_file.name if video_file else "",
        "frame_count": len(frames),
        "frame_observations": frame_observations,
        "summary": summary
    }


def normalize_file_stem(file_name):
    """
    统一文件名主干，用于把学生 .sb3 和运行视频进行粗略匹配。
    例如：张三.sb3、张三.mp4 可以自动匹配。
    """
    stem = os.path.splitext(os.path.basename(file_name or ""))[0]
    stem = re.sub(r"[\s_\-（）()]+", "", stem)
    return stem.lower()


def match_video_for_sb3(sb3_file_name, video_files):
    """
    根据文件名匹配对应运行视频。
    优先完全同名主干；如果只有一个视频，也默认匹配给当前作品。
    """
    if not video_files:
        return None

    if len(video_files) == 1:
        return video_files[0]

    sb3_stem = normalize_file_stem(sb3_file_name)
    for video_file in video_files:
        if normalize_file_stem(video_file.name) == sb3_stem:
            return video_file

    for video_file in video_files:
        video_stem = normalize_file_stem(video_file.name)
        if sb3_stem and (sb3_stem in video_stem or video_stem in sb3_stem):
            return video_file

    return None


# =========================
# 5. 调用 DeepSeek
# =========================

def build_api_messages(role: str, messages: list) -> list:
    role_prompt = build_role_prompt(role)

    api_messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + role_prompt}
    ]

    for msg in messages:
        api_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    return api_messages


def call_deepseek_full(role: str, messages: list) -> str:
    """
    非流式完整生成。教师端长回答使用这个函数，避免界面先显示半截内容。
    """
    api_messages = build_api_messages(role, messages)
    max_tokens = 12000 if role == "教师端" else 1600

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=api_messages,
        temperature=0.25,
        max_tokens=max_tokens,
        stream=False
    )

    raw_answer = response.choices[0].message.content or ""
    answer = postprocess_answer(role, raw_answer)

    finish_reason = getattr(response.choices[0], "finish_reason", "")
    if finish_reason == "length":
        answer += "\n\n提示：本次回答内容较长，可能仍有部分内容被模型长度限制截断。可以让智能体继续补充后续内容。"

    return answer


def stream_deepseek(role: str, messages: list, placeholder, scroll_holder=None) -> str:
    """
    学生端使用流式调用，让回答逐步出现。
    教师端建议使用 call_deepseek_full()，保证长教学设计完整显示。
    """
    api_messages = build_api_messages(role, messages)
    max_tokens = 1600 if role == "学生端" else 12000

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=api_messages,
        temperature=0.25,
        max_tokens=max_tokens,
        stream=True
    )

    full_answer = ""
    last_update_time = 0
    last_scroll_time = 0

    for chunk in response:
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:
            delta = ""

        if not delta:
            continue

        full_answer += delta
        now = time.time()

        if now - last_update_time > 0.08:
            # 流式中间态只做基础清理，不做编号重排，避免半句话被误处理成单独标题
            temp_answer = clean_answer(full_answer)
            placeholder.markdown(
                build_chat_bubble_html("assistant", temp_answer),
                unsafe_allow_html=True
            )
            last_update_time = now

        if now - last_scroll_time > 0.65:
            try:
                scroll_to_bottom(scroll_holder, smooth=False)
            except Exception:
                pass
            last_scroll_time = now

    answer = postprocess_answer(role, full_answer)

    # 最后必须用完整 full_answer 再渲染一次，避免界面停留在流式输出的中间状态。
    placeholder.markdown(
        build_chat_bubble_html("assistant", answer),
        unsafe_allow_html=True
    )

    try:
        scroll_to_bottom(scroll_holder, smooth=False)
    except Exception:
        pass

    return answer


# =========================
# 5.5 开场白问题
# =========================

def get_starter_questions(role: str):
    if role == "教师端":
        return [
            ("教学设计", "请帮我设计一节《打地鼠》图形化编程教学设计。"),
            ("任务单", "请帮我生成一份适合五年级学生的图形化编程任务单。"),
            ("调试支持", "请帮我整理学生常见的图形化编程调试问题。")
        ]

    return [
        ("任务分析", "我想做一个打地鼠小游戏，可以帮我分析一下怎么做吗？"),
        ("调试帮助", "我的角色只动了一次，应该先检查哪里？"),
        ("变量问题", "我想让分数增加，但分数没有变化，可以怎么排查？"),
        ("图片识别", "我上传了一张截图，请告诉我哪有问题。")
    ]


def set_quick_prompt(prompt_text: str):
    st.session_state.quick_prompt = prompt_text


# =========================
# 6. 侧边栏：身份入口
# =========================

st.sidebar.title("身份选择")

role_display_map = {"学生端": "学生", "教师端": "老师"}
user_role = st.sidebar.radio(
    "请选择你的身份",
    ["学生端", "教师端"],
    format_func=lambda role: role_display_map.get(role, role)
)


TEACHER_PASSWORD = get_secret_or_env("TEACHER_PASSWORD", "teacher123")

if user_role == "教师端":
    password = st.sidebar.text_input("请输入教师端密码", type="password")
    if password != TEACHER_PASSWORD:
        st.warning("老师身份需要密码。学生请选择左侧的“学生”。")
        st.stop()


# 身份切换时清空当前页面对话
if "current_role" not in st.session_state:
    st.session_state.current_role = user_role

if st.session_state.current_role != user_role:
    st.session_state.current_role = user_role
    st.session_state.messages = []
    st.session_state.quick_prompt = ""
    st.session_state.pasted_image = None


# 初始化聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []


# 初始化学生会话标识
if "student_session_id" not in st.session_state:
    st.session_state.student_session_id = ""


# 初始化快捷问题
if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = ""

if "last_sent_paste_hash" not in st.session_state:
    st.session_state.last_sent_paste_hash = ""


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
    col1, col2 = st.columns([1, 1])

    with col1:
        group_choice = st.selectbox(
            "小组号",
            ["第1组", "第2组", "第3组", "第4组", "第5组", "第6组", "第7组", "第8组", "第9组",  "第10组", "第11组", "第12组","其他小组"],
            index=0
        )

    with col2:
        topic_choice = st.selectbox(
            "当前主题",
            ["海底世界", "猫捉老鼠",  "牛顿的苹果","猜数字", "打地鼠",  "其他主题"],
            index=0
        )

    if group_choice == "其他小组":
        group_no = st.text_input("请输入小组号", placeholder="如：第9组")
    else:
        group_no = group_choice

    if topic_choice == "其他主题":
        topic = st.text_input("请输入当前主题", placeholder="如：海底世界")
    else:
        topic = topic_choice

    # 学生端不再要求填写姓名，减少小学生打字负担
    student_name = ""

    # 只根据“小组号 + 当前主题”判断是否开启新对话
    # 小组号或当前主题任意一个发生变化，就清空当前页面对话
    # 注意：这不会删除已经保存到 logs/chat_logs.csv 的历史记录
    current_student_session = f"{group_no.strip()}_{topic.strip()}"

    if group_no.strip() and topic.strip():
        if current_student_session != st.session_state.student_session_id:
            st.session_state.student_session_id = current_student_session
            st.session_state.messages = []
            st.session_state.quick_prompt = ""
            st.session_state.pasted_image = None

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
        "你好，我可以帮你分析图形化编程任务、检查程序问题，也可以识别你上传的程序截图。你可以直接提问，也可以从下面选一个问题开始。"
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
    render_chat_bubble(
        msg["role"],
        msg["content"],
        msg.get("image_base64", "")
    )

# 粘贴截图初始化：只负责暂存，不自动发送
if "pasted_image" not in st.session_state:
    st.session_state.pasted_image = None

send_pasted_only = False

# =========================
# 9. 聊天输入框：支持文字 + 图片附件 + 粘贴截图
# =========================

if user_role == "学生端":
    input_placeholder = "请输入问题，或点击附件上传图片。"
else:
    input_placeholder = "请输入教学需求，或点击附件上传图片。"

# 粘贴截图：只暂存，不自动发送
# 学生可以先粘贴图片，再在聊天框中输入问题发送；
# 也可以不输入问题，直接点击“发送截图”。

prompt = st.chat_input(
    input_placeholder,
    accept_file=True,
    file_type=["png", "jpg", "jpeg"]
)

# 固定在底部、靠近聊天输入框的粘贴截图工具条
# 保留原生 st.chat_input 的样式；截图按钮放在输入框下方同一底部区域。
if user_role in ["学生端", "教师端"]:
    with st.container(key="paste_toolbar"):
        # 这里始终定义 preview_col 和 clear_col，避免刚粘贴图片的这一轮中变量未定义。
        paste_col, send_img_col, preview_col, clear_col, tip_col = st.columns([0.8, 0.75, 0.28, 0.22, 4.9])

        with paste_col:
            pasted_result = paste_image_button(
                label="📋 粘贴截图",
                key="paste_image_button"
            )

        if pasted_result.image_data is not None:
            # paste_image_button 有时会在 rerun 后继续返回上一张图。
            # 用哈希避免“发送后清空了，又被组件自动塞回来”。
            try:
                _buf = BytesIO()
                pasted_result.image_data.save(_buf, format="PNG")
                _paste_hash = base64.b64encode(_buf.getvalue()).decode("utf-8")[:80]
            except Exception:
                _paste_hash = str(time.time())

            if _paste_hash != st.session_state.get("last_sent_paste_hash", ""):
                st.session_state.pasted_image = pasted_result.image_data
                st.session_state.current_paste_hash = _paste_hash

        with send_img_col:
            send_pasted_only = st.button(
                "发送截图",
                key="send_pasted_only",
                disabled=st.session_state.pasted_image is None
            )

        if st.session_state.pasted_image is not None:
            with preview_col:
                st.image(
                    st.session_state.pasted_image,
                    width=46
                )

            with clear_col:
                if st.button("×", key="clear_pasted_image"):
                    st.session_state.pasted_image = None
                    st.rerun()

            with tip_col:
                st.markdown(
                    '<div class="pending-image-box">已粘贴，可输入问题后发送，或直接发送截图。</div>',
                    unsafe_allow_html=True
                )
        else:
            with tip_col:
                st.empty()

if prompt:
    # Streamlit 版本不同，st.chat_input 可能返回字符串，也可能返回包含 text/files 的对象
    if isinstance(prompt, str):
        user_input = prompt
        uploaded_image = None
    else:
        user_input = getattr(prompt, "text", "") or ""
        uploaded_image = prompt.files[0] if getattr(prompt, "files", None) else None

elif send_pasted_only:
    user_input = ""
    uploaded_image = None

elif st.session_state.quick_prompt:
    user_input = st.session_state.quick_prompt
    uploaded_image = None
    st.session_state.quick_prompt = ""

else:
    user_input = None
    uploaded_image = None

# 本轮要发送的图片来源：
# ① 如果聊天框附件上传了图片，优先使用附件图片；
# ② 如果没有附件图片，但学生之前粘贴了截图，则随本轮文字一起发送粘贴截图；
# ③ 只粘贴截图但没有点击聊天框发送键时，不会触发智能体。
pasted_image_to_send = None
image_for_this_turn = uploaded_image

if user_input is not None and uploaded_image is None:
    if st.session_state.pasted_image is not None:
        pasted_image_to_send = st.session_state.pasted_image

if image_for_this_turn is None and pasted_image_to_send is not None:
    buffer = BytesIO()
    pasted_image_to_send.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = "pasted_image.png"
    buffer.type = "image/png"
    image_for_this_turn = buffer

# 只有真正输入文字、点击开场问题、聊天框发送文字并携带待发送截图，或点击“发送截图”时，才调用智能体
if user_input is not None:
    if user_role == "学生端":
        if not group_no.strip() or not topic.strip():
            st.warning("请先选择小组号和当前主题，再进行提问。")
            st.stop()

    display_user_input = user_input if user_input else "请识别这张图片中的文字。"
    api_user_input = user_input if user_input else "请识别这张图片中的文字，并结合小学图形化编程学习场景给出简要说明。"
    uploaded_image_name = ""
    uploaded_image_path = ""
    uploaded_image_base64 = ""
    ocr_text = ""

    # 如果上传了图片，先保存原图，再进行 OCR 文字识别
    if image_for_this_turn is not None:
        uploaded_image_name, uploaded_image_path = save_uploaded_image_file(image_for_this_turn)
        image_for_this_turn.seek(0)
        uploaded_image_base64 = base64.b64encode(image_for_this_turn.getvalue()).decode("utf-8")
        image_for_this_turn.seek(0)

        with st.spinner("正在识别图片中的文字，请稍等……"):
            ocr_text = recognize_image_text(image_for_this_turn)

        # 不把“已上传图片：xxx”重复显示在聊天文字里；
        # 图片会直接嵌入用户消息气泡中。
        api_user_input = f"""
【本轮用户最新问题】
{api_user_input}

【本轮截图文字识别结果】
{ocr_text}

请优先回答“本轮用户最新问题”，并结合“本轮截图文字识别结果”分析。

注意：
1. 只围绕本轮最新问题回答，不要重复回答历史问题。
2. 如果本轮有文字问题，必须先回应文字问题，再结合截图说明。
3. 不要把 OCR 结果逐字全部展示给学生，除非学生明确要求“识别文字”。
4. 如果识别结果不足以判断问题，请提醒用户上传更清晰的截图，或补充说明自己想实现什么、已经做了什么、出现了什么问题。
5. 面向学生时，仍然不能直接给完整程序，只能给分步提示或检查方向。
"""

    # 页面上显示用户文字和对应图片，历史消息中也保留这张图片
    st.session_state.messages.append({
        "role": "user",
        "content": display_user_input,
        "api_content": api_user_input,
        "image_base64": uploaded_image_base64
    })

    render_chat_bubble("user", display_user_input, uploaded_image_base64)

    # 用户消息出现后，尽量自动滚动到底部
    scroll_holder = st.empty()
    try:
        scroll_to_bottom(scroll_holder, smooth=True)
    except Exception:
        pass

    with st.spinner("正在生成中，请稍等……"):
        try:
            # 历史记录只放“用户真实显示文字”和助手回答，避免上一轮 OCR 长文本反复干扰本轮问题。
            # 本轮如果有截图，只把本轮 OCR 拼进最后一条 user 消息。
            messages_for_api = [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                for msg in st.session_state.messages[:-1]
            ] + [
                {
                    "role": "user",
                    "content": api_user_input
                }
            ]

            assistant_placeholder = st.empty()

            # 学生端保留流式输出；教师端使用完整生成，避免长教学设计在界面上显示半截。
            if user_role == "教师端":
                answer = call_deepseek_full(user_role, messages_for_api)
                assistant_placeholder.markdown(
                    build_chat_bubble_html("assistant", answer),
                    unsafe_allow_html=True
                )
            else:
                answer = stream_deepseek(
                    user_role,
                    messages_for_api,
                    assistant_placeholder,
                    scroll_holder
                )

            try:
                scroll_to_bottom(scroll_holder, smooth=True)
            except Exception:
                pass

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
                topic=topic,
                uploaded_image_name=uploaded_image_name,
                uploaded_image_path=uploaded_image_path,
                ocr_text=ocr_text,
                uploaded_image_base64=uploaded_image_base64
            )

            # 如果本轮使用了粘贴截图，发送后清空待发送区。
            # 发送后的图片已经进入聊天记录，不会消失。
            if pasted_image_to_send is not None:
                st.session_state.last_sent_paste_hash = st.session_state.get("current_paste_hash", "")
                st.session_state.pasted_image = None
                st.rerun()

        except Exception as e:
            st.error(f"调用失败：{e}")
# =========================
# 9.5 教师端：批量作品辅助评价
# =========================

if user_role == "教师端":
    st.markdown("### 批量作品辅助评价")

    with st.expander("上传学生作品和评价量表", expanded=False):
        rubric_file = st.file_uploader(
            "1.上传作品评价量表（txt / csv / xlsx）",
            type=["txt", "csv", "xlsx"],
            key="rubric_file"
        )
        reference_sb3_file = st.file_uploader(
            "2.上传教师基础版作品 .sb3（用于提供核心功能目标参照，可选但推荐）",
            type=["sb3"],
            key="reference_sb3_file"
        )

        reference_video_file = st.file_uploader(
            "3.上传教师基础版作品运行视频（用于提供运行效果参照，可选但推荐）",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            key="reference_video_file"
        )

        sb3_files = st.file_uploader(
            "4.批量上传学生 .sb3 作品文件",
            type=["sb3"],
            accept_multiple_files=True,
            key="sb3_files"
        )

        video_files = st.file_uploader(
            "5.可选：批量上传学生作品运行视频",
            type=["mp4", "mov", "avi", "mkv", "webm"],
            accept_multiple_files=True,
            key="video_files"
        )


        st.caption("教师基础版视频用于提供标准运行效果参照；学生视频建议与 .sb3 文件同名，例如：张三.sb3 和 张三.mp4。若只上传一个学生视频，会默认匹配到当前作品。")

        start_batch_eval = st.button("开始批量辅助评价")

        if start_batch_eval:
            if rubric_file is None:
                st.warning("请先上传作品评价量表。")
                st.stop()

            if not sb3_files:
                st.warning("请至少上传一个 .sb3 作品文件。")
                st.stop()

            # 读取评价量表
            rubric_text = ""

            if rubric_file.name.endswith(".txt"):
                rubric_text = rubric_file.getvalue().decode("utf-8", errors="ignore")

            elif rubric_file.name.endswith(".csv"):
                df_rubric = pd.read_csv(rubric_file)
                rubric_text = df_rubric.to_string(index=False)

            elif rubric_file.name.endswith(".xlsx"):
                df_rubric = pd.read_excel(rubric_file)
                rubric_text = df_rubric.to_string(index=False)

            # 读取教师基础版作品。它只作为“核心功能目标参照”，不是标准答案代码。
            reference_analysis = None
            if reference_sb3_file is not None:
                reference_analysis = analyze_sb3_file(reference_sb3_file)

            # 分析教师基础版运行视频。它用于提供当前任务的可见运行效果参照。
            # 这里用 st.info / st.success 保留提示，避免 spinner 完成后提示消失，造成“没有分析教师视频”的误解。
            reference_video_analysis = None
            if reference_video_file is not None:
                st.info(f"正在分析教师基础版运行视频：{reference_video_file.name}")
                with st.spinner(f"正在分析教师基础版运行视频：{reference_video_file.name}"):
                    reference_video_analysis = analyze_project_video_with_qwen_vl(
                        reference_video_file,
                        task_context=(rubric_text + "\n这是教师基础版作品运行视频，用于说明本节课作品应呈现的基础运行效果和核心交互。"),
                    )
                st.success(f"教师基础版运行视频分析完成：{reference_video_file.name}")
            else:
                st.warning("未上传教师基础版运行视频，本次评价将主要参考教师基础版 .sb3、学生作品、学生视频和评价量表。")

            results = []

            with st.spinner("正在批量解析学生作品并生成辅助评价，请稍等……"):
                for sb3_file in sb3_files:
                    analysis = analyze_sb3_file(sb3_file)

                    matched_video = match_video_for_sb3(sb3_file.name, video_files)
                    video_analysis = None
                    if matched_video is not None:
                        st.info(f"正在分析学生作品运行视频：{matched_video.name}")
                        video_analysis = analyze_project_video_with_qwen_vl(
                            matched_video,
                            task_context=(
                                rubric_text
                                + "\n这是学生作品运行视频，请观察学生作品实际运行效果。"
                                + ("\n教师基础版运行视频分析参照：" + json.dumps(reference_video_analysis, ensure_ascii=False) if reference_video_analysis else "")
                            ),
                        )

                    eval_text = evaluate_project_with_rubric(
                        sb3_analysis=analysis,
                        rubric_text=rubric_text,
                        reference_analysis=reference_analysis,
                        video_analysis=video_analysis,
                        reference_video_analysis=reference_video_analysis
                    )

                    dimension_scores = parse_dimension_evaluations(eval_text)
                    running_analysis = extract_eval_field(eval_text, "运行画面分析")
                    if not running_analysis and video_analysis:
                        running_analysis = video_analysis.get("summary", "")
                    if not running_analysis:
                        running_analysis = "未上传运行视频，本次评价主要依据作品结构分析、教师基础作品和评价量表。"

                    row_result = {
                        "作品文件": sb3_file.name,
                        "教师基础视频": reference_video_file.name if reference_video_file is not None else "",
                        "匹配视频": matched_video.name if matched_video is not None else "",
                        "视频关键帧数": video_analysis.get("frame_count", "") if video_analysis else "",
                        "运行画面分析": running_analysis,
                        "核心功能完成情况": extract_eval_field(eval_text, "核心功能完成情况"),
                        "完成等级": normalize_completion_level(extract_eval_field(eval_text, "完成等级")),
                    }

                    # 根据教师上传的评价量表动态生成“各维度评分及依据”列
                    row_result.update(dimension_scores)

                    row_result.update({
                        "建议总分": extract_score_number(extract_eval_field(eval_text, "建议总分")),
                        "总评依据": extract_eval_field(eval_text, "总评依据") or extract_eval_field(eval_text, "评分依据"),
                        "改进建议": extract_eval_field(eval_text, "改进建议"),
                        "教师复核点": extract_eval_field(eval_text, "教师复核点"),
                        "视频逐帧观察": json.dumps(video_analysis.get("frame_observations", []), ensure_ascii=False) if video_analysis else "",

                        "角色数量": analysis.get("sprite_count", ""),
                        "角色名称": "、".join(analysis.get("sprites", [])),
                        "积木总数": analysis.get("block_count", ""),
                        "事件积木": analysis.get("event_blocks", ""),
                        "循环积木": analysis.get("loop_blocks", ""),
                        "条件积木": analysis.get("condition_blocks", ""),
                        "变量积木": analysis.get("variable_blocks", ""),
                        "广播积木": analysis.get("broadcast_blocks", ""),
                        "结构分析": analysis.get("summary", "")
                    })

                    results.append(row_result)
            result_df = pd.DataFrame(results)

            # 动态调整列顺序：基础信息 → 运行与核心功能 → 动态维度评分及依据 → 总评与复核 → 结构分析
            fixed_front_cols = [
                "作品文件", "教师基础视频", "匹配视频", "视频关键帧数", "运行画面分析",
                "核心功能完成情况", "完成等级"
            ]
            fixed_back_cols = [
                "建议总分", "总评依据", "改进建议", "教师复核点", "视频逐帧观察",
                "角色数量", "角色名称", "积木总数", "事件积木", "循环积木",
                "条件积木", "变量积木", "广播积木", "结构分析"
            ]
            dynamic_dimension_cols = [
                col for col in result_df.columns
                if col.endswith("评分及依据")
                and col not in fixed_front_cols
                and col not in fixed_back_cols
            ]
            ordered_cols = (
                [col for col in fixed_front_cols if col in result_df.columns]
                + dynamic_dimension_cols
                + [col for col in fixed_back_cols if col in result_df.columns]
            )
            other_cols = [col for col in result_df.columns if col not in ordered_cols]
            result_df = result_df[ordered_cols + other_cols]

            st.success("批量辅助评价已完成。")
            st.dataframe(result_df, use_container_width=True)

            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                result_df.to_excel(writer, index=False, sheet_name="辅助评价结果")

            st.download_button(
                label="下载批量评价结果 Excel",
                data=output.getvalue(),
                file_name="批量作品辅助评价结果.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
# =========================
# 10. 侧边栏：教师端操作区
# =========================

# 教师端只保留一个下载按钮：下载全部记录 Excel（含上传图片）
if user_role == "教师端":
    logs = load_logs()

    if not logs.empty:
        st.sidebar.divider()
        st.sidebar.markdown("### 对话记录")

        excel_data = create_excel_with_images(logs)
        st.sidebar.download_button(
            label="下载全部对话记录",
            data=excel_data,
            file_name="chat_logs_with_images.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.sidebar.divider()
        st.sidebar.markdown("### 对话记录")
        st.sidebar.info("暂无可下载的对话记录。")
