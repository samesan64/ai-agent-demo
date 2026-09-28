"""
アクセス制御と使用回数の管理(Streamlitに依存しない部分)
------------------------------------------------------
ai_agent_demo.py から使われる部品です。画面(Streamlit)とは切り離してあるので、
単体でテストできます(test_access_control.py を参照)。

■ 役割
  1. verify_code : 入力されたアクセスコードが正しいか照合する
  2. UsageStore  : コードごとの使用回数をファイルに記録し、上限を超えないように管理する

■ 注意(保存先について)
  使用回数はJSONファイルに保存しています。Streamlit Community Cloudでは、
  アプリの再起動や再デプロイでファイルが消えることがあり、その場合は回数がリセットされます。
  本格的に運用する場合は、UsageStore と同じメソッド(get_used / try_consume / refund)を持つ
  クラスを外部データベース向けに作り、差し替えてください。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import threading
from dataclasses import dataclass

ADMIN_KIND = "admin"
BUYER_KIND = "buyer"


@dataclass(frozen=True)
class AccessInfo:
    """ログインに成功した人の情報"""
    kind: str            # "admin"(自分用・回数無制限) または "buyer"(購入者)
    code_id: str         # 回数記録用のID(コードそのものではなく、ハッシュ値から作った文字列)
    limit: int | None    # 使用回数の上限。None は無制限


def make_code_id(code: str) -> str:
    """
    アクセスコードから、記録用のIDを作る。
    使用回数のファイルにコードそのものを書かないよう、ハッシュ化した値の先頭部分を使う。
    """
    return hashlib.sha256(code.encode("utf-8")).hexdigest()[:16]


def verify_code(entered: str, access_codes: dict, admin_code: str = "") -> AccessInfo | None:
    """
    入力されたコードを照合する。
    - admin_code と一致 → 管理者(回数無制限)
    - access_codes(コード → 使用回数の上限)のどれかと一致 → 購入者
    - どれとも一致しない → None

    文字列の比較には hmac.compare_digest を使う。通常の == だと、
    一致する文字数によって処理時間がわずかに変わり、そこから推測される危険があるため。
    """
    entered = (entered or "").strip()
    if not entered:
        return None
    entered_bytes = entered.encode("utf-8")

    result: AccessInfo | None = None

    if admin_code and hmac.compare_digest(entered_bytes, str(admin_code).strip().encode("utf-8")):
        result = AccessInfo(ADMIN_KIND, make_code_id(entered), None)

    # 途中で打ち切らず、すべてのコードと比較する(比較にかかる時間を一定に近づけるため)
    for code, limit in access_codes.items():
        if hmac.compare_digest(entered_bytes, str(code).strip().encode("utf-8")):
            try:
                limit_value = int(limit)
            except (TypeError, ValueError):
                continue  # 上限の書き方が不正なコードは無効として扱う
            if result is None:
                result = AccessInfo(BUYER_KIND, make_code_id(entered), limit_value)

    return result


class UsageStore:
    """
    コードごとの使用回数を、JSONファイルに保存して管理する。

    - try_consume: 「上限に達していないか確認」と「1回分増やす」を、ひとまとまりの処理として行う
                   (別々にやると、同時に2人が使ったとき上限を超えてしまう可能性があるため)
    - 書き込みは「一時ファイルに書いてから置き換える」方式で、書き込み中に落ちてもファイルが壊れにくい
    - 排他制御(Lock)は、同じプロセス内の同時アクセスにだけ有効
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            # ファイルが壊れていた場合は、原因調査用に退避して、空の状態から始める
            try:
                os.replace(self.path, self.path + ".corrupt")
            except OSError:
                pass
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict) -> None:
        directory = os.path.dirname(os.path.abspath(self.path))
        fd, tmp_path = tempfile.mkstemp(dir=directory, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp_path, self.path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def get_used(self, code_id: str) -> int:
        """これまでに使った回数を返す"""
        with self._lock:
            return int(self._read().get(code_id, 0))

    def try_consume(self, code_id: str, limit: int) -> bool:
        """上限に達していなければ、使用回数を1つ増やして True を返す。達していれば増やさず False を返す。"""
        with self._lock:
            data = self._read()
            used = int(data.get(code_id, 0))
            if used >= limit:
                return False
            data[code_id] = used + 1
            self._write(data)
            return True

    def refund(self, code_id: str) -> None:
        """使用回数を1つ戻す(処理が失敗したとき、購入者が損をしないようにするため)"""
        with self._lock:
            data = self._read()
            used = int(data.get(code_id, 0))
            if used > 0:
                data[code_id] = used - 1
                self._write(data)
