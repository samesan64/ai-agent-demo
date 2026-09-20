"""
AI Agent デモツール(Web版・Streamlit使用)
------------------------------------
今までのツールとの違い:
  今まで  → 「①要約→②整理」のように、"あなた"が処理の順番を決めていた
  今回    → "AI自身"が、依頼内容を見て「どの道具(ツール)を使うべきか」を判断する

Claude APIの「Tool Use(関数呼び出し)」という仕組みを使い、
AIに複数の道具(Pythonの関数)を持たせて、自分で選んで使わせます。
これは実際の「AI Agent」開発の基本的な仕組みそのものです。

■ 事前準備
pip install streamlit anthropic

■ 起動方法
python -m streamlit run ai_agent_demo.py
"""

import ast
import operator
import os

import streamlit as st
from anthropic import Anthropic
import anthropic

# ---- APIキー設定 ----
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
if not API_KEY:
    # secrets.tomlファイル自体が存在しない環境では st.secrets へのアクセスで例外が出ることがあるため、
    # 環境変数に無い場合だけ試し、失敗しても静かに空文字のまま続行する。
    try:
        API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        API_KEY = ""
if not API_KEY:
    st.error("APIキーが設定されていません。環境変数またはSecretsにANTHROPIC_API_KEYを登録してください。")
    st.stop()

client = Anthropic(api_key=API_KEY)


# ---- 設定値(まとめておくことで、後から調整しやすくする) ----
MODEL_NAME = "claude-sonnet-4-6"
PRICE_INPUT_PER_MILLION_USD = 3     # Claude Sonnetの入力料金の目安(1トークン=1/100万ドル単位)
PRICE_OUTPUT_PER_MILLION_USD = 15   # Claude Sonnetの出力料金の目安
USD_TO_JPY_RATE = 150               # 大まかな円換算レート(実際の為替レートにより変動)
MAX_AGENT_LOOP_STEPS = 5            # 1回の依頼につき、道具を使ってよい最大の往復回数
MAX_API_RETRIES = 3                 # API呼び出しが失敗した時の再試行回数
MAX_UPLOAD_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 添付ファイルの上限サイズ(5MB)。大きすぎるファイルはコスト・処理時間の暴走を招くため制限する


# =========================================================
# ① AIに持たせる「道具(ツール)」たち
#    それぞれ、ただのPythonの関数です。
# =========================================================

def tool_summarize(text: str) -> str:
    """文章を1〜2文に要約する道具"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=[{"role": "user", "content": f"次の文章を1〜2文で要約してください。要約のみ出力:\n{text}"}],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def tool_count_words(text: str) -> str:
    """文字数・単語数を数える道具(AIを使わず、Pythonだけで計算)"""
    char_count = len(text)
    word_count = len(text.split())
    return f"文字数: {char_count}文字 / 単語数(空白区切り): {word_count}語"


def tool_extract_keywords(text: str) -> str:
    """重要なキーワードを抽出する道具"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=200,
        messages=[{"role": "user", "content": f"次の文章から重要なキーワードを5個まで、カンマ区切りで抽出してください。キーワードのみ出力:\n{text}"}],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def tool_translate(text: str, target_language: str) -> str:
    """指定した言語に翻訳する道具(英語だけでなく、複数の言語に対応)"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=400,
        messages=[{"role": "user", "content": f"次の文章を{target_language}に翻訳してください。翻訳文のみ出力してください:\n{text}"}],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def tool_get_today() -> str:
    """今日の日付を取得する道具(AIは「今日が何日か」を正確には知らないので、Pythonから教える)"""
    import datetime
    today = datetime.date.today()
    weekday_jp = ["月", "火", "水", "木", "金", "土", "日"][today.weekday()]
    return f"今日は{today.year}年{today.month}月{today.day}日({weekday_jp}曜日)です。"


def tool_analyze_sentiment(text: str) -> str:
    """文章の感情(ポジティブ/ネガティブ/ニュートラル)を分析する道具"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"次の文章の感情を分析してください。\n"
                f"「ポジティブ」「ネガティブ」「ニュートラル」のいずれかで判定し、"
                f"その理由を1文で簡潔に説明してください。\n"
                f"出力形式: 【判定】理由\n\n"
                f"---\n{text}\n---"
            ),
        }],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def tool_extract_action_items(text: str) -> str:
    """会議メモや長文から「やるべきこと(アクションアイテム)」を抽出する道具"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                f"次の文章から、実行すべきタスク(アクションアイテム)を抽出してください。\n"
                f"分かる範囲で「担当者」「内容」「期限」を箇条書きでまとめてください。"
                f"担当者や期限が書かれていない場合は「未定」としてください。\n\n"
                f"---\n{text}\n---"
            ),
        }],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def tool_draft_email(purpose: str, recipient: str, tone: str = "丁寧") -> str:
    """ビジネスメールの文面を自動生成する道具"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": (
                f"以下の条件で、日本語のビジネスメールの文面を作成してください。\n"
                f"宛先(相手): {recipient}\n"
                f"メールの目的: {purpose}\n"
                f"文体: {tone}\n"
                f"件名と本文を含めて出力してください。前置きの説明は不要です。"
            ),
        }],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def tool_plan_steps(goal: str) -> str:
    """大きな目標・漠然とした依頼を、実行可能な手順に分解する道具"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": (
                f"次の目標を達成するために必要な手順を、実行しやすい順番の"
                f"ステップに分解してください。各ステップは短く、具体的にしてください。\n\n"
                f"目標: {goal}"
            ),
        }],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def tool_compare_texts(text_a: str, text_b: str) -> str:
    """2つの文章を比較し、違いと共通点をまとめる道具"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": (
                f"次の2つの文章A・Bを比較してください。\n"
                f"「共通点」「Aだけにある内容」「Bだけにある内容」に分けて、"
                f"箇条書きで簡潔にまとめてください。\n\n"
                f"【文章A】\n{text_a}\n\n【文章B】\n{text_b}"
            ),
        }],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def tool_draft_sns_post(topic: str, platform: str = "Twitter(X)") -> str:
    """SNS投稿文の下書きを作成する道具"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": (
                f"{platform}に投稿する文章を作成してください。\n"
                f"内容・トピック: {topic}\n"
                f"そのSNSらしい、読みやすく親しみやすいトーンで、2〜3パターン作成してください。"
            ),
        }],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def tool_proofread(text: str) -> str:
    """文章の誤字脱字・表現のおかしな点をチェックする道具"""
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": (
                f"次の文章を校正してください。誤字脱字、不自然な言い回し、"
                f"文法的な誤りを指摘し、修正案を示してください。\n"
                f"問題が見つからない場合は「問題は見つかりませんでした」と答えてください。\n\n"
                f"---\n{text}\n---"
            ),
        }],
    )
    return "".join(b.text for b in res.content if b.type == "text")


# 計算式の中で許可する演算子だけを列挙する(これ以外は一切実行できない)
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_CALC_NUMBER = 10 ** 12  # これより大きい数値・累乗結果は拒否する(暴走計算を防ぐ)
_MAX_POW_EXPONENT = 20       # 指数(累乗の右側)の上限。9**9**9のような桁数爆発を防ぐ


def _safe_eval_node(node):
    """
    計算式のASTノードを再帰的に評価する。
    eval()と違い、四則演算・カッコ・数値以外は一切実行できないので、
    変数名や関数呼び出し(__import__など)を使った攻撃を受け付けない。
    """
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("数値以外の値が含まれています。")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"「{op_type.__name__}」という演算子は使えません。")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError(f"累乗の指数が大きすぎます(上限: {_MAX_POW_EXPONENT})。")
        result = _ALLOWED_OPERATORS[op_type](left, right)
        if isinstance(result, (int, float)) and abs(result) > _MAX_CALC_NUMBER:
            raise ValueError("計算結果の桁数が大きすぎるため、安全のため計算を中止しました。")
        return result

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_OPERATORS:
            raise ValueError(f"「{op_type.__name__}」という演算子は使えません。")
        return _ALLOWED_OPERATORS[op_type](_safe_eval_node(node.operand))

    raise ValueError("数式として認識できない要素が含まれています。")


def tool_calculate(expression: str) -> str:
    """
    正確な計算を行う道具。
    AI(言語モデル)は、桁数の大きい計算や複雑な計算を間違えることがあるため、
    計算が必要な場面では、AI自身に計算させず、Pythonに正確に計算させる。

    安全のため eval() は使わず、数式をASTに変換したうえで
    四則演算・カッコ・数値以外のノードを一切評価しない「安全な電卓」として実装している。
    """
    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval_node(parsed)
        return f"{expression} = {result}"
    except ZeroDivisionError:
        return "計算エラー: 0で割ることはできません。"
    except (SyntaxError, ValueError) as e:
        return f"計算エラー: {e}"
    except Exception as e:
        return f"計算エラー: 計算式を解析できませんでした({e})"


def tool_extract_contact_info(text: str) -> str:
    """
    文章の中から、メールアドレス・電話番号・URLを抜き出す道具。
    AIを使わず、Pythonの「正規表現(regex)」というパターンマッチングの技術だけで動く。
    データ入力・情報整理の実務でよく使われる処理。
    """
    import re

    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phones = re.findall(r"0\d{1,4}-\d{1,4}-\d{3,4}", text)
    urls = re.findall(r"https?://[^\s]+", text)

    lines = []
    lines.append(f"📧 メールアドレス({len(emails)}件): " + (", ".join(emails) if emails else "見つかりませんでした"))
    lines.append(f"📞 電話番号({len(phones)}件): " + (", ".join(phones) if phones else "見つかりませんでした"))
    lines.append(f"🔗 URL({len(urls)}件): " + (", ".join(urls) if urls else "見つかりませんでした"))
    return "\n".join(lines)


def tool_save_note(key: str, value: str) -> str:
    """
    情報を「長期記憶」として保存する道具。
    通常の会話履歴と違い、この記憶はチャットをリセットしても消えない
    (AI自身が「これは覚えておくべきだ」と判断した時に使う)。
    """
    if "agent_notes" not in st.session_state:
        st.session_state.agent_notes = {}
    st.session_state.agent_notes[key] = value
    return f"「{key}」として「{value}」を記憶しました。"


def tool_recall_notes(query: str = "") -> str:
    """記憶しているメモを一覧で呼び出す道具"""
    notes = st.session_state.get("agent_notes", {})
    if not notes:
        return "まだ何も記憶していません。"
    lines = [f"・{k}: {v}" for k, v in notes.items()]
    return "記憶している内容:\n" + "\n".join(lines)


def tool_ask_user(question: str, reason: str = "") -> str:
    """
    情報が足りなくて判断できない時に、ユーザーに質問する道具。
    推測で勝手に進めるのではなく、きちんと確認を取るために使う。
    """
    result = f"❓ ユーザーへの確認が必要です。\n質問: {question}"
    if reason:
        result += f"\n理由: {reason}"
    result += "\n\n(この質問をユーザーに伝え、回答を待ってください。推測で進めないこと。)"
    return result


def tool_create_chart(chart_type: str, labels: list, values: list, title: str = "", y_label: str = "") -> str:
    """
    数値データからグラフ(画像)を作る道具。
    matplotlibで描画し、画面に表示できるよう session_state に保存する。
    """
    import matplotlib
    matplotlib.use("Agg")  # 画面を持たない環境でも描画できるモードにする
    import matplotlib.pyplot as plt
    import japanize_matplotlib  # 日本語が文字化けしないようにする
    import io

    # 入力チェック:ラベルと数値の個数が合っているか
    if len(labels) != len(values):
        return f"エラー: 項目名({len(labels)}個)と数値({len(values)}個)の数が一致しません。"

    if not labels or not values:
        return "エラー: 項目名・数値が空です。1件以上のデータを指定してください。"

    if chart_type not in ("bar", "line", "pie"):
        return f"エラー: 未対応のグラフ種類「{chart_type}」です。bar / line / pie のいずれかを指定してください。"

    # 数値であるべき項目に文字列などが混ざっていないかチェックする
    cleaned_values = []
    for i, v in enumerate(values):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return f"エラー: {i + 1}番目の数値「{v}」が数値として認識できません。"
        cleaned_values.append(float(v))
    values = cleaned_values

    try:
        fig, ax = plt.subplots(figsize=(8, 5))

        if chart_type == "bar":
            ax.bar(labels, values, color="#4A90D9")
        elif chart_type == "line":
            ax.plot(labels, values, marker="o", color="#4A90D9", linewidth=2)
        elif chart_type == "pie":
            if any(v < 0 for v in values):
                plt.close(fig)
                return "エラー: 円グラフ(pie)にはマイナスの数値を使えません。"
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")

        if title:
            ax.set_title(title, fontsize=14)
        if y_label and chart_type != "pie":
            ax.set_ylabel(y_label)
        if chart_type in ("bar", "line"):
            plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        # 画像をメモリ上に保存して、画面表示用に取っておく
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)
    except Exception as e:
        plt.close("all")
        return f"エラー: グラフの作成中に問題が発生しました({e})"

    if "generated_charts" not in st.session_state:
        st.session_state.generated_charts = []
    st.session_state.generated_charts.append(buf.getvalue())

    return f"✅ {chart_type}グラフを作成しました(タイトル: {title or 'なし'}、項目数: {len(labels)})。画面に表示されます。"


# 道具の実体(名前で呼び出せるようにする辞書)
AVAILABLE_TOOLS = {
    "summarize": tool_summarize,
    "count_words": tool_count_words,
    "extract_keywords": tool_extract_keywords,
    "translate": tool_translate,
    "get_today": tool_get_today,
    "analyze_sentiment": tool_analyze_sentiment,
    "extract_action_items": tool_extract_action_items,
    "draft_email": tool_draft_email,
    "plan_steps": tool_plan_steps,
    "compare_texts": tool_compare_texts,
    "draft_sns_post": tool_draft_sns_post,
    "proofread": tool_proofread,
    "calculate": tool_calculate,
    "extract_contact_info": tool_extract_contact_info,
    "save_note": tool_save_note,
    "recall_notes": tool_recall_notes,
    "ask_user": tool_ask_user,
    "create_chart": tool_create_chart,
}

# ---------------------------------------------------------
# ② Claude APIに「こんな道具があります」と伝えるための説明書(スキーマ)
#    AIはこれを読んで、どの道具を使うか自分で判断します。
# ---------------------------------------------------------
TOOL_DEFINITIONS = [
    {
        "name": "summarize",
        "description": "長い文章を1〜2文に要約する。要約してほしい、短くまとめてほしい、という依頼のときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要約したい文章"}},
            "required": ["text"],
        },
    },
    {
        "name": "count_words",
        "description": "文章の文字数・単語数を数える。文字数を知りたい、というときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "文字数を数えたい文章"}},
            "required": ["text"],
        },
    },
    {
        "name": "extract_keywords",
        "description": "文章から重要なキーワードを抽出する。キーワードを知りたい、というときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "キーワードを抽出したい文章"}},
            "required": ["text"],
        },
    },
    {
        "name": "translate",
        "description": "文章を指定された言語に翻訳する。「〜語に翻訳して」「英訳して」「中国語にして」などの依頼のときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "翻訳したい文章"},
                "target_language": {
                    "type": "string",
                    "description": "翻訳先の言語名(例: 英語, 中国語, 韓国語, フランス語 など)。ユーザーの依頼から判断して指定する。",
                },
            },
            "required": ["text", "target_language"],
        },
    },
    {
        "name": "get_today",
        "description": "今日の日付・曜日を正確に取得する。今日の日付を知りたい、という依頼のときに使う。引数は不要。",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "analyze_sentiment",
        "description": "文章の感情(ポジティブ・ネガティブ・ニュートラル)を分析する。感情分析してほしい、この文章はポジティブかネガティブか知りたい、というときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "感情分析したい文章"}},
            "required": ["text"],
        },
    },
    {
        "name": "extract_action_items",
        "description": "会議メモや長文から、実行すべきタスク(誰が・何を・いつまでに)を抽出する。やることリストを作りたい、アクションアイテムを知りたい、というときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "タスクを抽出したい文章(会議メモなど)"}},
            "required": ["text"],
        },
    },
    {
        "name": "draft_email",
        "description": "ビジネスメールの文面を作成する。メールを書いて、メール文を作って、という依頼のときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "description": "メールの目的(何を伝えたいか)"},
                "recipient": {"type": "string", "description": "宛先・相手(例: 取引先, 上司, お客様 など)"},
                "tone": {"type": "string", "description": "文体(例: 丁寧, カジュアル, フォーマル)。指定がなければ「丁寧」"},
            },
            "required": ["purpose", "recipient"],
        },
    },
    {
        "name": "plan_steps",
        "description": "大きな目標や漠然とした依頼を、実行可能な手順(ステップ)に分解する。計画を立てて、何からやればいい、手順を教えて、というときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {"goal": {"type": "string", "description": "達成したい目標・やりたいこと"}},
            "required": ["goal"],
        },
    },
    {
        "name": "compare_texts",
        "description": "2つの文章を比較し、共通点・それぞれだけの違いをまとめる。2つを比較して、違いを教えて、というときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {
                "text_a": {"type": "string", "description": "比較対象の文章A"},
                "text_b": {"type": "string", "description": "比較対象の文章B"},
            },
            "required": ["text_a", "text_b"],
        },
    },
    {
        "name": "draft_sns_post",
        "description": "SNS(Twitter/Instagramなど)向けの投稿文を作成する。SNSに投稿する文章を作って、という依頼のときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "投稿したい内容・トピック"},
                "platform": {"type": "string", "description": "投稿先のSNS名(例: Twitter(X), Instagram, TikTok)。指定がなければTwitter(X)"},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "proofread",
        "description": "文章の誤字脱字・不自然な表現・文法ミスをチェックする。校正して、誤字がないか確認して、というときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "校正したい文章"}},
            "required": ["text"],
        },
    },
    {
        "name": "calculate",
        "description": "正確な計算を行う(足し算・引き算・掛け算・割り算)。計算して、合計はいくつ、というように数値計算が必要なときに必ず使う。AI自身の暗算に頼らず、必ずこの道具を使うこと。",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "計算式(例: 1200*3+500)"}},
            "required": ["expression"],
        },
    },
    {
        "name": "extract_contact_info",
        "description": "文章の中からメールアドレス・電話番号・URLを抜き出す。連絡先を抽出して、メアドある?、というときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "連絡先情報を探したい文章"}},
            "required": ["text"],
        },
    },
    {
        "name": "save_note",
        "description": "重要な情報を長期記憶として保存する。ユーザーが「覚えておいて」と言った時や、後で使いそうな重要な情報が出てきた時に、自発的に使う。",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "何についてのメモか(短いラベル、例: 好きな食べ物)"},
                "value": {"type": "string", "description": "記憶する内容"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall_notes",
        "description": "これまでに記憶したメモを一覧で呼び出す。何を覚えてる?、というときに使う。",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "(任意)特に知りたいことがあれば"}},
            "required": [],
        },
    },
    {
        "name": "ask_user",
        "description": "依頼を実行するのに必要な情報が足りない時、推測で進めずにユーザーに質問する。例えばメール作成で宛先が不明な場合や、依頼の意図が曖昧な場合に使う。",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "ユーザーに聞きたいこと"},
                "reason": {"type": "string", "description": "なぜその情報が必要なのか"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "create_chart",
        "description": "数値データからグラフを作成する。グラフにして、可視化して、図で見せて、という依頼のときに使う。データの性質に応じて、棒グラフ(bar)・折れ線(line)・円グラフ(pie)を自分で選ぶこと。",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "pie"],
                    "description": "グラフの種類。比較はbar、推移はline、割合はpie",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "各データの項目名(例: ['1月','2月','3月'])",
                },
                "values": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "各項目に対応する数値(labelsと同じ個数)",
                },
                "title": {"type": "string", "description": "グラフのタイトル"},
                "y_label": {"type": "string", "description": "縦軸のラベル(単位など)"},
            },
            "required": ["chart_type", "labels", "values"],
        },
    },
]


# ---------------------------------------------------------
# 道具のカテゴリ分け
# 道具が増えると、毎回すべての説明をAIに送るのが無駄になる(料金も増える)。
# カテゴリごとに絞り込めるようにして、必要な道具だけ渡せるようにする。
# ---------------------------------------------------------
TOOL_CATEGORIES = {
    "📝 文章の分析": ["summarize", "count_words", "extract_keywords", "analyze_sentiment", "compare_texts", "proofread"],
    "✍️ 文章の作成": ["draft_email", "draft_sns_post", "translate"],
    "📋 業務サポート": ["extract_action_items", "plan_steps", "extract_contact_info"],
    "🧠 記憶・確認": ["save_note", "recall_notes", "ask_user"],
    "🔢 計算・その他": ["calculate", "get_today"],
    "📊 データ可視化": ["create_chart"],
}


def get_filtered_tools(selected_categories: list) -> list:
    """選択されたカテゴリに属する道具だけを取り出す"""
    if not selected_categories:
        return TOOL_DEFINITIONS  # 何も選ばれていなければ全部使う

    allowed_names = set()
    for category in selected_categories:
        allowed_names.update(TOOL_CATEGORIES.get(category, []))

    return [tool for tool in TOOL_DEFINITIONS if tool["name"] in allowed_names]



# 「もう一度試せば直る可能性が高い」= 一時的なエラーとみなす例外クラス
_RETRYABLE_EXCEPTIONS = (
    anthropic.RateLimitError,       # 429: リクエストが多すぎる
    anthropic.APIConnectionError,   # 通信エラー・タイムアウト
    anthropic.InternalServerError,  # 500系: サーバー側の一時的な問題(529の混雑含む)
)


def call_claude_with_retry(create_kwargs: dict, max_retries: int = MAX_API_RETRIES):
    """
    Claude APIを呼び出す。一時的なエラー(混雑・通信エラーなど)が起きた場合、
    少し待ってから自動で再試行する(最大 MAX_API_RETRIES 回まで)。

    エラーの種類は、文字列の中身を見て推測するのではなく、
    anthropicライブラリが定義している例外クラス(RateLimitErrorなど)で正確に判定する。
    これにより「一時的なエラーではないのに誤って再試行してしまう」ことを防ぐ。
    """
    import time

    for attempt in range(max_retries):
        try:
            return client.messages.create(**create_kwargs)
        except _RETRYABLE_EXCEPTIONS as e:
            if attempt < max_retries - 1:
                wait_seconds = 2 ** attempt  # 1回目は1秒待つ、2回目は2秒、3回目は4秒(待ち時間を徐々に伸ばす)
                st.toast(f"⏳ 混雑しているようです。{wait_seconds}秒後に再試行します…({attempt + 1}/{max_retries})")
                time.sleep(wait_seconds)
                continue
            raise  # 再試行回数を使い切った場合は、エラーをそのまま発生させる
        except anthropic.APIStatusError:
            # 400番台のクライアントエラー(リクエスト内容の誤りなど)は再試行しても直らないため、即座にエラーを返す
            raise


def reflect_and_improve(user_request: str, draft_answer: str) -> str:
    """
    AIが一度出した回答を、もう一度自分で見直して改善する「セルフチェック」の仕組み。
    (Reflection と呼ばれる、AI Agent開発の定番テクニック)
    """
    res = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": (
                f"以下は、ある依頼に対してAIが出した回答の下書きです。\n"
                f"依頼内容と照らし合わせて、抜け漏れ・誤り・分かりにくい点がないか確認し、"
                f"必要であれば改善してください。問題がなければ、そのままの内容を出力してください。\n"
                f"改善後の回答のみを出力し、チェック過程の説明は不要です。\n\n"
                f"【元の依頼】\n{user_request}\n\n"
                f"【回答の下書き】\n{draft_answer}"
            ),
        }],
    )
    return "".join(b.text for b in res.content if b.type == "text")


def run_agent_step(messages: list, log: list, system_prompt: str = "", budget_jpy: float = 0, active_tools: list = None) -> str:
    """
    AI Agentのメイン処理(会話履歴を引き継ぐ版)。
    active_tools: 今回使わせる道具のリスト(Noneなら全部)
    """
    if active_tools is None:
        active_tools = TOOL_DEFINITIONS
    if "total_input_tokens" not in st.session_state:
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0

    for _ in range(MAX_AGENT_LOOP_STEPS):
        # 予算チェック:上限に達していたら、APIを呼ぶ前に処理を止める
        if budget_jpy > 0:
            current_cost_usd = (
                st.session_state.total_input_tokens / 1_000_000 * PRICE_INPUT_PER_MILLION_USD
                + st.session_state.total_output_tokens / 1_000_000 * PRICE_OUTPUT_PER_MILLION_USD
            )
            current_cost_jpy = current_cost_usd * USD_TO_JPY_RATE
            if current_cost_jpy >= budget_jpy:
                log.append(f"🛑 予算上限(約{budget_jpy}円)に達したため、処理を停止しました。")
                return f"⚠️ 設定した予算の上限(約{budget_jpy}円)に達したため、これ以上の処理を停止しました。「会話をリセット」するか、予算を見直してください。"

        create_kwargs = {
            "model": MODEL_NAME,
            "max_tokens": 2000,
            "tools": active_tools,
            "messages": messages,
        }
        if system_prompt.strip():
            create_kwargs["system"] = system_prompt

        response = call_claude_with_retry(create_kwargs)

        # 今回のAPI呼び出しで使ったトークン数を記録する
        st.session_state.total_input_tokens += response.usage.input_tokens
        st.session_state.total_output_tokens += response.usage.output_tokens

        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            messages.append({"role": "assistant", "content": response.content})
            return final_text

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                log.append(f"🔧 「{tool_name}」を実行(入力: {str(tool_input)[:40]}...)")
                func = AVAILABLE_TOOLS.get(tool_name)
                is_error = False
                if func is None:
                    result_text = f"エラー: 道具 '{tool_name}' は存在しません"
                    is_error = True
                else:
                    try:
                        result_text = func(**tool_input)
                    except TypeError as e:
                        # 引数の過不足など、AIが渡したパラメータの形が想定と違う場合
                        result_text = f"エラー: 「{tool_name}」の呼び出しに失敗しました(引数の形式が不正です: {e})"
                        is_error = True
                    except Exception as e:
                        # 道具の実行中に何が起きても、ここで受け止めて会話を継続できるようにする。
                        # ここで捕まえないと、tool_useに対応するtool_resultが返らないまま
                        # 例外が外側まで伝播し、次回のAPI呼び出しで会話履歴が不正な形になってしまう。
                        result_text = f"エラー: 「{tool_name}」の実行中に問題が発生しました({e})"
                        is_error = True
                if is_error:
                    log.append(f"⚠️ 「{tool_name}」の実行でエラー: {result_text}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                    "is_error": is_error,
                })
        messages.append({"role": "user", "content": tool_results})

    return "処理が複雑すぎたため、途中で打ち切りました。"


# =========================================================
# ③ 画面(UI)部分 ― チャット形式に変更
#    会話履歴は st.session_state に保存し、ページを操作しても消えないようにする
# =========================================================
st.set_page_config(page_title="AI Agent チャット", page_icon="🤖")

st.title("🤖 AI Agent チャット")
st.write("会話を続けながら、AIが必要な道具を自分で選んで実行します。前のやり取りも覚えています。")

# 初めて使う人向けのガイド(まだ会話が始まっていない時だけ表示)
if not st.session_state.get("messages"):
    st.info(
        "**はじめての方へ** \n\n"
        "下の入力欄に、やってほしいことを普通の言葉で書くだけでOKです。\n"
        "AIが「どの機能を使うべきか」を自分で判断して実行します。"
    )

    st.write("**こんなことができます(クリックで試せます)**")
    col1, col2 = st.columns(2)
    example_prompts = [
        "今日は何日?",
        "取引先に納期遅延をお詫びするメールを書いて",
        "1280円のランチを月22日食べたらいくら?",
        "AI開発で副業を始めたい。何から始めればいい?",
    ]
    for i, example in enumerate(example_prompts):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            if st.button(example, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_example = example
                st.rerun()

with st.expander("🔧 AIが使える道具一覧"):
    st.write("- **summarize**: 文章を要約する")
    st.write("- **count_words**: 文字数・単語数を数える")
    st.write("- **extract_keywords**: キーワードを抽出する")
    st.write("- **translate**: 指定した言語に翻訳する(英語・中国語・韓国語など、なんでも可)")
    st.write("- **get_today**: 今日の日付・曜日を取得する")
    st.write("- **analyze_sentiment**: 文章の感情(ポジティブ/ネガティブ/ニュートラル)を分析する")
    st.write("- **extract_action_items**: 会議メモなどから「やること」を抽出する")
    st.write("- **draft_email**: ビジネスメールの文面を作成する")
    st.write("- **plan_steps**: 大きな目標を実行可能な手順に分解する")
    st.write("- **compare_texts**: 2つの文章を比較して違い・共通点をまとめる")
    st.write("- **draft_sns_post**: SNS投稿文の下書きを作成する")
    st.write("- **proofread**: 誤字脱字・不自然な表現をチェックする")
    st.write("- **calculate**: 正確な計算をする(AIの暗算ミスを防ぐ)")
    st.write("- **extract_contact_info**: メールアドレス・電話番号・URLを抜き出す")
    st.write("- **save_note / recall_notes**: 重要な情報を長期記憶として保存・呼び出しする")
    st.write("- **ask_user**: 情報が足りない時、推測せずにユーザーに質問する")
    st.write("- **create_chart**: 数値データからグラフ(棒・折れ線・円)を作成する")

with st.expander("⚙️ AIの役割・性格を設定する(システムプロンプト)"):
    st.write("AI全体の振る舞いのルールを、自由に設定できます。空欄なら標準の振る舞いになります。")
    system_prompt = st.text_area(
        "例: あなたは親しみやすい後輩キャラです。語尾に「〜っす」をつけて話してください。",
        value=st.session_state.get("system_prompt", ""),
        height=80,
        key="system_prompt",
    )

with st.expander("💰 予算の上限を設定する(使いすぎ防止)"):
    st.write("このセッションで使ってよい上限金額を設定できます。0円なら無制限です。")
    budget_jpy = st.number_input("上限金額(円)", min_value=0, value=0, step=10, key="budget_jpy")

enable_reflection = st.checkbox(
    "🔍 セルフチェックを有効にする(回答の質は上がりますが、料金も増えます)",
    value=False,
)

selected_categories = st.multiselect(
    "🗂️ 使う道具のカテゴリを絞る(未選択なら全部使います。絞ると料金の節約になります)",
    options=list(TOOL_CATEGORIES.keys()),
    default=[],
)
active_tools = get_filtered_tools(selected_categories)
if selected_categories:
    st.caption(f"現在 {len(active_tools)}個の道具が有効です(全{len(TOOL_DEFINITIONS)}個中)")

# 会話履歴を保存する場所(初回だけ空リストを作る)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 利用料金の目安を表示する(Claude Sonnetの料金: 入力$3/100万トークン, 出力$15/100万トークン で概算)
if st.session_state.get("total_input_tokens", 0) > 0:
    input_tokens = st.session_state.total_input_tokens
    output_tokens = st.session_state.total_output_tokens
    cost_usd = (input_tokens / 1_000_000 * PRICE_INPUT_PER_MILLION_USD) + (
        output_tokens / 1_000_000 * PRICE_OUTPUT_PER_MILLION_USD
    )
    cost_jpy = cost_usd * USD_TO_JPY_RATE  # 大まかな円換算(為替レートにより変動)

    with st.expander(f"💰 このセッションの利用料金の目安: 約{cost_jpy:.2f}円"):
        st.write(f"入力トークン: {input_tokens:,} / 出力トークン: {output_tokens:,}")
        st.caption("※ Claude Sonnetの料金をもとにした概算です。実際の請求額とは為替レート等により差が生じます。")

# リセットボタン(新しい話題に切り替えたいとき用)
if st.button("🔄 会話をリセット"):
    st.session_state.messages = []
    st.session_state.total_input_tokens = 0
    st.session_state.total_output_tokens = 0
    st.session_state.tool_logs = {}
    st.rerun()

# ファイルアップロード(.txt / .pdf)
uploaded_file = st.file_uploader("📎 ファイルを添付する(.txt / .pdf)", type=["txt", "pdf"])

# 「このファイルは、次の1回のメッセージにだけ添付する」という状態を管理する
if "pending_attachment" not in st.session_state:
    st.session_state.pending_attachment = None

if uploaded_file is not None:
    # ファイルサイズの上限チェック(大きすぎるファイルは処理時間・トークン費用の暴走につながる)
    if uploaded_file.size > MAX_UPLOAD_FILE_SIZE_BYTES:
        st.error(
            f"⚠️ ファイルサイズが大きすぎます({uploaded_file.size / 1_000_000:.1f}MB)。"
            f"上限は{MAX_UPLOAD_FILE_SIZE_BYTES / 1_000_000:.0f}MBです。"
        )
    # 同じファイルを何度も読み込み直さないよう、ファイル名で重複チェック
    elif st.session_state.get("last_uploaded_name") != uploaded_file.name:
        try:
            if uploaded_file.name.endswith(".pdf"):
                from pypdf import PdfReader
                reader = PdfReader(uploaded_file)
                page_texts = []
                for page_number, page in enumerate(reader.pages, start=1):
                    try:
                        page_texts.append(page.extract_text() or "")
                    except Exception:
                        # 1ページだけ読み込みに失敗しても、他のページは読み込みを続ける
                        page_texts.append(f"[{page_number}ページ目の読み込みに失敗しました]")
                file_text = "\n".join(page_texts)
            else:
                file_text = uploaded_file.read().decode("utf-8", errors="ignore")

            st.session_state.pending_attachment = file_text
            st.session_state.last_uploaded_name = uploaded_file.name
            st.info(f"📄 「{uploaded_file.name}」を読み込みました({len(file_text)}文字)。次のメッセージに1回だけ添付されます。")
        except Exception as e:
            st.error(f"⚠️ ファイルの読み込みに失敗しました: {e}")

# 会話をファイルに保存する機能(ポートフォリオ用の証拠としても使える)
if st.session_state.messages:
    export_lines = ["# AI Agent 会話ログ\n"]
    for msg in st.session_state.messages:
        if msg["role"] == "user" and isinstance(msg["content"], str):
            export_lines.append(f"**あなた:** {msg['content']}\n")
        elif msg["role"] == "assistant":
            text_parts = [b.text for b in msg["content"] if getattr(b, "type", None) == "text" and b.text.strip()]
            if text_parts:
                export_lines.append(f"**AI Agent:**\n{chr(10).join(text_parts)}\n")
    export_content = "\n".join(export_lines)

    st.download_button(
        "💾 この会話をファイルで保存する",
        data=export_content,
        file_name="ai_agent_conversation.md",
        mime="text/markdown",
    )

# これまでの会話を画面に表示する(道具の実行過程は隠し、最終回答だけ見せる)
for idx, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user" and isinstance(msg["content"], str):
        with st.chat_message("user"):
            st.write(msg["content"])
    elif msg["role"] == "assistant":
        text_parts = [b.text for b in msg["content"] if getattr(b, "type", None) == "text" and b.text.strip()]
        if text_parts:
            with st.chat_message("assistant"):
                # その時に使った道具があれば、上に小さく表示する
                tool_log = st.session_state.get("tool_logs", {}).get(idx)
                if tool_log:
                    tool_names = [entry.split("「")[1].split("」")[0] for entry in tool_log if "「" in entry]
                    if tool_names:
                        st.caption("🔧 使用した道具: " + " → ".join(tool_names))
                answer_text = "\n".join(text_parts)
                st.markdown(answer_text)
                st.download_button(
                    "💾 この回答だけ保存",
                    data=answer_text,
                    file_name="ai_answer.md",
                    mime="text/markdown",
                    key=f"download_history_{idx}",
                )

# 新しいメッセージの入力欄(チャット形式)
user_input = st.chat_input("依頼を入力してください(例: この文章を要約して/日本語の要約を英語にもして)")

# サンプルボタンが押された場合、それを入力として扱う
if st.session_state.get("pending_example"):
    user_input = st.session_state.pending_example
    st.session_state.pending_example = None

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    # 添付ファイルがあれば、依頼文の後ろに自動でくっつけてAIに渡す(1回使ったら消す)
    message_for_ai = user_input
    if st.session_state.pending_attachment:
        message_for_ai += f"\n\n【添付ファイルの内容】\n{st.session_state.pending_attachment}"
        st.session_state.pending_attachment = None  # 使い終わったので消す(次回からは添付されない)

    st.session_state.messages.append({"role": "user", "content": message_for_ai})

    log = []
    st.session_state.generated_charts = []  # 今回の依頼で作るグラフを入れる場所をリセット
    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            try:
                final_answer = run_agent_step(st.session_state.messages, log, system_prompt, budget_jpy, active_tools)

                if enable_reflection:
                    with st.spinner("🔍 回答を見直しています..."):
                        final_answer = reflect_and_improve(user_input, final_answer)

                # 使った道具の記録を、会話履歴と紐づけて保存する(後から見返せるように)
                if "tool_logs" not in st.session_state:
                    st.session_state.tool_logs = {}
                if log:
                    st.session_state.tool_logs[len(st.session_state.messages)] = log
                    tool_names = [entry.split("「")[1].split("」")[0] for entry in log if "「" in entry]
                    if tool_names:
                        st.caption("🔧 使用した道具: " + " → ".join(tool_names))

                st.markdown(final_answer)

                # グラフが作られていれば表示する
                for chart_bytes in st.session_state.get("generated_charts", []):
                    st.image(chart_bytes)
                    st.download_button(
                        "📊 グラフを画像で保存",
                        data=chart_bytes,
                        file_name="chart.png",
                        mime="image/png",
                        key=f"chart_dl_{len(st.session_state.messages)}_{id(chart_bytes)}",
                    )

                # この回答だけを個別に保存できるようにする
                st.download_button(
                    "💾 この回答だけ保存",
                    data=final_answer,
                    file_name="ai_answer.md",
                    mime="text/markdown",
                    key=f"download_latest_{len(st.session_state.messages)}",
                )
            except anthropic.AuthenticationError:
                st.error("⚠️ APIキーが正しくないか、無効になっています。設定を確認してください。")
            except anthropic.RateLimitError:
                st.error("⚠️ APIの利用制限(レート制限)に達しました。しばらく時間をおいてから再度お試しください。")
            except anthropic.APIConnectionError:
                st.error("⚠️ APIへの接続に失敗しました。通信環境を確認し、もう一度お試しください。")
            except Exception as e:
                st.error(f"⚠️ エラーが発生しました: {e}")
