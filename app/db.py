"""SQLite 存储层。

同步驱动 sqlite3，每请求一个连接。金额全程以整数分存储与读取
（core 层金额也是整数分，落库不再做任何元↔分浮点换算），
刷卡时刻以 'YYYY-MM-DD HH:MM:SS' 文本存储（可直接排序）。
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

from .core import Config, Fare, Station, Swipe, yuan_of_cents

DB_PATH = os.environ.get("DB_PATH", "/data/bus.db")

TIME_FMT = "%Y-%m-%d %H:%M:%S"


def _ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _yuan(cents: int) -> float:
    return yuan_of_cents(int(cents))


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS station (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name  TEXT NOT NULL UNIQUE,
    zone  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS fare (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    from_zone  INTEGER NOT NULL,
    to_zone    INTEGER NOT NULL,
    price_cents INTEGER NOT NULL,
    UNIQUE(from_zone, to_zone)
);

CREATE TABLE IF NOT EXISTS swipe (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    card     TEXT NOT NULL,
    t        TEXT NOT NULL,
    station  TEXT NOT NULL,
    action   TEXT NOT NULL,        -- IN / OUT
    src_line INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_swipe_card_t ON swipe(card, t);

CREATE TABLE IF NOT EXISTS trip_bill (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    card             TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    operating_day    TEXT NOT NULL,
    entry_time       TEXT NOT NULL,
    exit_time        TEXT NOT NULL,
    entry_station    TEXT NOT NULL,
    exit_station     TEXT NOT NULL,
    from_zone        INTEGER NOT NULL,
    to_zone          INTEGER NOT NULL,
    base_cents       INTEGER NOT NULL,
    discount_cents   INTEGER NOT NULL,
    charge_cents     INTEGER NOT NULL,
    cumulative_cents INTEGER NOT NULL,
    cap_cents        INTEGER NOT NULL,
    discount_source  TEXT NOT NULL,
    entry_line       INTEGER NOT NULL,
    exit_line        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bill_card ON trip_bill(card, seq);
"""

# 规则参数持久化键：金额一律以整数分文本保存（与 core 的整数分口径一致）。
DEFAULT_CONFIG = {
    "transfer_minutes": "90",
    "transfer_discount_cents": "100",
    "daily_cap_cents": "1000",
    "day_split_hour": "4",
}

# 数据版本（meta 表键）：站点分区、票价矩阵、刷卡记录与规则参数
# 每次保存都在同一事务内递增该版本并使账单失效；核验据此发现
# 「读取快照之后数据已被其他请求修改」的并发冲突。版本持久化在
# 数据库中，跨进程生效；旧数据库启动时由 init_db 自动补齐。
DATA_VERSION_KEY = "data_version"


@contextmanager
def connect(db_path: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    path = db_path or DB_PATH
    _ensure_parent(path)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # 写锁冲突时短暂等待而非立即报错：核验提交与保存操作可能跨进程并发
    conn.execute("PRAGMA busy_timeout = 5000")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Optional[str] = None) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        for k, v in DEFAULT_CONFIG.items():
            conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)", (k, v))
        # 数据版本：旧数据库启动时自动补齐；已存在则保留当前值
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES (?, ?)",
            (DATA_VERSION_KEY, "0"))


# ---------------------------------------------------------------------------
# 在线备份（只读：不修改任何业务数据）
# ---------------------------------------------------------------------------

# 备份时每次复制的页数与页间休眠（秒）：分块进行可在备份期间
# 让并发写入继续提交，SQLite 会重传被改动的页，保证目标文件
# 始终对应源库某一个完整提交点。
_BACKUP_PAGES = 128
_BACKUP_SLEEP = 0.05


def create_backup_snapshot(dest_path: str, db_path: Optional[str] = None) -> None:
    """把当前数据库的一致性快照写入 dest_path（SQLite 在线备份 API）。

    - 源库以只读 URI（mode=ro）打开，备份全程只执行读取，
      不修改任何业务数据；
    - 备份期间若有并发写入，在线备份会重传被改动的页，目标文件
      始终对应源库某一个完整提交点（事务边界），绝不直接复制
      正在使用的数据库文件；
    - 写入完成后对目标文件执行 PRAGMA integrity_check，
      未通过则抛出 RuntimeError（调用方负责清理临时文件）。
    """
    src_path = db_path or DB_PATH
    _ensure_parent(dest_path)
    uri = Path(src_path).absolute().as_uri() + "?mode=ro"
    src = sqlite3.connect(uri, uri=True)
    try:
        dst = sqlite3.connect(dest_path)
        try:
            src.backup(dst, pages=_BACKUP_PAGES, sleep=_BACKUP_SLEEP)
        finally:
            dst.close()
    finally:
        src.close()

    check = sqlite3.connect(dest_path)
    try:
        row = check.execute("PRAGMA integrity_check").fetchone()
    finally:
        check.close()
    if not row or row[0] != "ok":
        raise RuntimeError(
            f"备份文件完整性检查未通过：{row[0] if row else '无结果'}")


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value))


# ---------------------------------------------------------------------------
# 数据版本（并发核验保护）
# ---------------------------------------------------------------------------

def get_data_version(conn: sqlite3.Connection) -> int:
    """当前数据版本：站点 / 票价 / 刷卡记录 / 配置的累计保存次数。"""
    v = get_meta(conn, DATA_VERSION_KEY)
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def bump_data_version(conn: sqlite3.Connection) -> int:
    """在同一事务内递增数据版本（保存操作的调用方负责同事务使账单失效）。"""
    v = get_data_version(conn) + 1
    set_meta(conn, DATA_VERSION_KEY, str(v))
    return v


def begin_snapshot(conn: sqlite3.Connection) -> None:
    """开启显式读事务：其后的多次读取来自同一一致快照。

    只持有读（SHARED）锁，读取结束应立即 rollback / commit 释放，
    避免长时间阻塞并发写入。
    """
    conn.execute("BEGIN")


def begin_write(conn: sqlite3.Connection) -> None:
    """开启显式写事务并立即获取写锁（BEGIN IMMEDIATE）。

    版本复查与账单写入在同一事务内完成：事务提交前，其他连接的
    写事务无法介入，因此「版本未变」的结论在提交时仍然成立，
    该保护跨进程生效。
    """
    conn.execute("BEGIN IMMEDIATE")


def _parse_config_cents(value: Optional[str], default: str) -> int:
    """读取以整数分持久化的金额配置。

    兼容旧版本以「元」文本（如 ``"1.00"``）保存的配置：含小数点时
    按十进制元换算为分；旧键缺失时回退默认值（同样为分文本）。
    """
    if value is None:
        value = default
    text = str(value).strip()
    if "." in text:
        from decimal import Decimal, InvalidOperation
        try:
            return int(Decimal(text).scaleb(2))
        except (InvalidOperation, ValueError):
            return int(default)
    try:
        return int(text)
    except ValueError:
        return int(default)


def get_config(conn: sqlite3.Connection) -> Config:
    def g(key: str) -> Optional[str]:
        return get_meta(conn, key)

    transfer_minutes = g("transfer_minutes")
    day_split_hour = g("day_split_hour")

    # 金额（整数分）。旧版本以「元」文本键 transfer_discount / daily_cap
    # 保存；只要旧键仍存在（首次保存新配置时会被删除），就以旧键为准
    # 按十进制换算为分，保证旧库升级后生效的仍是用户配置而非默认值。
    def amount_cents(new_key: str, legacy_key: str, default_key: str) -> int:
        legacy = g(legacy_key)
        if legacy is not None:
            return _parse_config_cents(legacy, DEFAULT_CONFIG[default_key])
        return _parse_config_cents(g(new_key), DEFAULT_CONFIG[default_key])

    return Config(
        transfer_minutes=int(
            transfer_minutes if transfer_minutes is not None
            else DEFAULT_CONFIG["transfer_minutes"]),
        transfer_discount_cents=amount_cents(
            "transfer_discount_cents", "transfer_discount",
            "transfer_discount_cents"),
        daily_cap_cents=amount_cents(
            "daily_cap_cents", "daily_cap", "daily_cap_cents"),
        day_split_hour=int(
            day_split_hour if day_split_hour is not None
            else DEFAULT_CONFIG["day_split_hour"]),
    )


def update_config(conn: sqlite3.Connection, cfg: Config) -> None:
    set_meta(conn, "transfer_minutes", str(cfg.transfer_minutes))
    set_meta(conn, "transfer_discount_cents",
             str(cfg.transfer_discount_cents))
    set_meta(conn, "daily_cap_cents", str(cfg.daily_cap_cents))
    set_meta(conn, "day_split_hour", str(cfg.day_split_hour))
    # 清理旧版本的元文本金额键，避免两套键并存
    conn.execute("DELETE FROM meta WHERE key IN ('transfer_discount', 'daily_cap')")


# ---------------------------------------------------------------------------
# 基础数据：整表替换（原子事务，错误时由调用方不调用）
# ---------------------------------------------------------------------------

def replace_stations(conn: sqlite3.Connection, stations: list[Station]) -> None:
    conn.execute("DELETE FROM station")
    conn.executemany(
        "INSERT INTO station(name, zone) VALUES (?, ?)",
        [(s.name, s.zone) for s in stations])


def replace_fares(conn: sqlite3.Connection, fares: list[Fare]) -> None:
    conn.execute("DELETE FROM fare")
    conn.executemany(
        "INSERT INTO fare(from_zone, to_zone, price_cents) VALUES (?, ?, ?)",
        [(f.from_zone, f.to_zone, f.price_cents) for f in fares])


def replace_records(conn: sqlite3.Connection, records: list[Swipe]) -> None:
    conn.execute("DELETE FROM swipe")
    conn.executemany(
        "INSERT INTO swipe(card, t, station, action, src_line) "
        "VALUES (?, ?, ?, ?, ?)",
        [(r.card, r.t.strftime(TIME_FMT), r.station, r.action, r.line)
         for r in records])


def save_bills(conn: sqlite3.Connection, bills: list[dict]) -> None:
    """一次核验产出：先清空旧账单，再写入新账单（同一事务）。

    账单金额直接使用核验计算出的整数分，与核验同一口径，
    不再经任何浮点换算，边界值不会溢出或失真。
    """
    conn.execute("DELETE FROM trip_bill")
    conn.executemany(
        """INSERT INTO trip_bill(card, seq, operating_day, entry_time, exit_time,
               entry_station, exit_station, from_zone, to_zone,
               base_cents, discount_cents, charge_cents, cumulative_cents,
               cap_cents, discount_source, entry_line, exit_line)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [(b["card"], b["seq"], b["operating_day"], b["entry_time"],
          b["exit_time"], b["entry_station"], b["exit_station"],
          b["from_zone"], b["to_zone"],
          b["base_fare_cents"], b["discount_cents"],
          b["charged_fare_cents"], b["daily_cumulative_cents"],
          b["daily_cap_cents"], b["discount_source"],
          b["entry_line"], b["exit_line"]) for b in bills])
    set_meta(conn, "last_verified_at",
             datetime.now().strftime(TIME_FMT))


def clear_bills(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM trip_bill")
    conn.execute("DELETE FROM meta WHERE key='last_verified_at'")


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

def list_stations(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, zone FROM station ORDER BY zone, name").fetchall()
    return [dict(r) for r in rows]


def list_fares(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, from_zone, to_zone, price_cents "
        "FROM fare ORDER BY from_zone, to_zone").fetchall()
    return [{"id": r["id"], "from_zone": r["from_zone"],
             "to_zone": r["to_zone"], "price": _yuan(r["price_cents"])}
            for r in rows]


def list_records(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT id, card, t, station, action, src_line "
        "FROM swipe ORDER BY id").fetchall()
    return [{"id": r["id"], "card": r["card"], "time": r["t"],
             "station": r["station"], "action": r["action"],
             "src_line": r["src_line"]} for r in rows]


def load_domain(conn: sqlite3.Connection):
    """读出 core 层所需的全部输入数据。"""
    stations = [Station(name=r["name"], zone=r["zone"])
                for r in conn.execute("SELECT name, zone FROM station")]
    fares = [Fare(from_zone=r["from_zone"], to_zone=r["to_zone"],
                  price_cents=r["price_cents"])
             for r in conn.execute(
                 "SELECT from_zone, to_zone, price_cents FROM fare")]
    records: list[Swipe] = []
    for r in conn.execute(
            "SELECT card, t, station, action, src_line FROM swipe ORDER BY id"):
        records.append(Swipe(
            card=r["card"],
            t=datetime.strptime(r["t"], TIME_FMT),
            station=r["station"], action=r["action"], line=r["src_line"]))
    return stations, fares, records


def list_bills(conn: sqlite3.Connection) -> list[dict]:
    """读出最近一次核验账单。金额以整数分键返回（与 core 账单同一口径）；
    HTTP 输出边界（main.serialize_bills）再换算为元。"""
    rows = conn.execute(
        """SELECT * FROM trip_bill ORDER BY card, seq, id""").fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r["id"], "card": r["card"], "seq": r["seq"],
            "operating_day": r["operating_day"],
            "entry_time": r["entry_time"], "exit_time": r["exit_time"],
            "entry_station": r["entry_station"],
            "exit_station": r["exit_station"],
            "from_zone": r["from_zone"], "to_zone": r["to_zone"],
            "base_fare_cents": r["base_cents"],
            "discount_cents": r["discount_cents"],
            "charged_fare_cents": r["charge_cents"],
            "daily_cumulative_cents": r["cumulative_cents"],
            "daily_cap_cents": r["cap_cents"],
            "discount_source": r["discount_source"],
            "entry_line": r["entry_line"], "exit_line": r["exit_line"],
        })
    return out
