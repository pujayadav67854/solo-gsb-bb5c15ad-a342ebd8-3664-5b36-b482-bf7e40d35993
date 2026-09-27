"""并发核验回归测试：数据版本保护（可控制交错顺序）。

通过 monkeypatch 替换 ``main.verify`` 计算函数，在「读取一致快照」与
「提交账单」之间精确插入另一请求的保存操作，确定性复现并发交错：

1. 旧核验读入 2 元票价后，另一请求保存 5 元票价并清空账单，
   旧核验不得写回 2 元账单（返回 409 + DATA_CHANGED）；
2. 过期的失败核验不得清掉另一请求已生成的新账单；
3. 各保存接口在同一事务内递增数据版本并使账单失效；
4. 无冲突时保留原有成功 / 422 行为，核验本身不递增版本；
5. 旧数据库启动时自动补齐数据版本（且不清空已有版本）。

运行：``python3 -m pytest tests/ -q``（依赖见 requirements.txt，
另需 pytest 与 httpx）。
"""
from __future__ import annotations

import json

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app import core, db, main
from app.schemas import ConfigIn, TextIn

STATIONS = "科技园, 1\n中心公园, 2\n"
FARES_2 = "1, 1, 2.00\n1, 2, 3.00\n2, 1, 3.00\n2, 2, 2.00\n"
FARES_5 = "1, 1, 5.00\n1, 2, 5.00\n2, 1, 5.00\n2, 2, 5.00\n"
RECORDS_OK = (
    "A001, 2026-09-18 08:00:00, 科技园, 进\n"
    "A001, 2026-09-18 08:25:00, 中心公园, 出\n"
)
# 缺出站：核验必然失败（MISSING_EXIT）
RECORDS_BAD = "A001, 2026-09-18 08:00:00, 科技园, 进\n"


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """每个用例一个独立数据库文件（db.connect 默认读取模块级 DB_PATH）。"""
    path = str(tmp_path / "bus-test.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    return path


def _seed(fares: str = FARES_2, records: str = RECORDS_OK) -> None:
    """通过保存接口写入基础数据（每次保存都会递增数据版本）。"""
    main.put_stations(TextIn(text=STATIONS))
    main.put_fares(TextIn(text=fares))
    main.put_records(TextIn(text=records))


def _payload(resp: JSONResponse) -> dict:
    return json.loads(resp.body.decode("utf-8"))


def _state() -> tuple[list[dict], str | None, int]:
    """当前 (账单, 核验时间, 数据版本)。"""
    with db.connect() as conn:
        return (db.list_bills(conn),
                db.get_meta(conn, "last_verified_at"),
                db.get_data_version(conn))


def _interleave_once(monkeypatch, hook) -> None:
    """把 main.verify 包一层：仅在第一次核验计算期间执行 hook。

    hook 在「快照已读取、结果未提交」的窗口内运行，等效于另一请求
    恰好在此刻并发保存数据；第二次起（如重新核验）直接调用原函数。
    """
    real_verify = core.verify
    fired = []

    def spy(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        if not fired:
            fired.append(True)
            hook()
        return result

    monkeypatch.setattr(main, "verify", spy)


# ---------------------------------------------------------------------------
# 冲突场景：过期核验不得覆盖新数据
# ---------------------------------------------------------------------------

def test_stale_success_verify_rejected_and_not_written(temp_db, monkeypatch):
    """读入 2 元票价后另一请求保存 5 元票价：旧核验 409，不写回 2 元账单。"""
    _seed()
    first = main.run_verify()
    assert not isinstance(first, JSONResponse)  # 首次核验成功
    bills, verified_at, v0 = _state()
    assert len(bills) == 1 and bills[0]["base_fare"] == 3.0
    assert verified_at is not None

    def concurrent_save():
        # 另一请求：保存 5 元票价（同一事务内递增版本并使账单失效）
        main.put_fares(TextIn(text=FARES_5))

    _interleave_once(monkeypatch, concurrent_save)
    resp = main.run_verify()

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 409
    payload = _payload(resp)
    assert payload["ok"] is False
    assert payload["data_changed"] is True
    assert payload["errors"][0]["code"] == "DATA_CHANGED"

    bills, verified_at, v1 = _state()
    # 旧核验未写回 2 元账单：账单仍是保存票价时清空的状态
    assert bills == []
    assert verified_at is None
    assert v1 == v0 + 1  # 仅保存票价递增了一次，核验本身不递增

    # 重新核验：基于新数据正常出账（5 元），无冲突时行为不变
    again = main.run_verify()
    assert not isinstance(again, JSONResponse)
    assert again["ok"] is True
    assert again["bills"][0]["base_fare"] == 5.0


def test_stale_failed_verify_does_not_clear_new_bills(temp_db, monkeypatch):
    """过期失败核验不得清掉另一请求已生成的新账单。"""
    _seed(records=RECORDS_BAD)  # 缺出站：本次核验将失败

    def concurrent_fix_and_verify():
        # 另一请求：修复记录并成功核验，生成新账单
        main.put_records(TextIn(text=RECORDS_OK))
        inner = main.run_verify()
        assert not isinstance(inner, JSONResponse)
        assert inner["ok"] is True

    _interleave_once(monkeypatch, concurrent_fix_and_verify)
    resp = main.run_verify()

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 409
    assert _payload(resp)["errors"][0]["code"] == "DATA_CHANGED"

    bills, verified_at, _v = _state()
    # 新账单与核验时间未被过期失败核验清掉
    assert len(bills) == 1
    assert bills[0]["charged_fare"] == 3.0
    assert verified_at is not None


def test_stale_verify_rejected_over_http(temp_db, monkeypatch):
    """端到端：冲突经 HTTP 返回 409 + DATA_CHANGED，账单保持最新状态。"""
    _seed()
    client = TestClient(main.app)
    assert client.post("/api/verify").status_code == 200

    _interleave_once(monkeypatch,
                     lambda: main.put_fares(TextIn(text=FARES_5)))
    resp = client.post("/api/verify")
    assert resp.status_code == 409
    body = resp.json()
    assert body["ok"] is False
    assert body["data_changed"] is True
    assert body["errors"][0]["code"] == "DATA_CHANGED"

    current = client.get("/api/bills").json()
    assert current["count"] == 0
    assert current["last_verified_at"] is None


def test_cross_connection_interleaving(temp_db):
    """跨连接（等效跨进程）交错：写事务内能发现另一连接已提交的版本变化。"""
    _seed()
    # 连接 A：一致快照读取输入与版本
    with db.connect() as conn_a:
        db.begin_snapshot(conn_a)
        version_a = db.get_data_version(conn_a)
        snapshot = db.load_domain(conn_a)
        conn_a.rollback()
    assert len(snapshot[0]) == 2  # 快照读到 2 个站点

    # 连接 B（等效另一进程）：保存新票价，同一事务内递增版本并使账单失效
    with db.connect() as conn_b:
        db.replace_fares(conn_b, [core.Fare(1, 1, 5.0)])
        db.bump_data_version(conn_b)
        db.clear_bills(conn_b)

    # 连接 A 的写事务：版本已变化，必须放弃提交
    with db.connect() as conn_a2:
        db.begin_write(conn_a2)
        assert db.get_data_version(conn_a2) != version_a
        conn_a2.rollback()


# ---------------------------------------------------------------------------
# 保存操作：同一事务内更新数据、递增版本并使账单失效
# ---------------------------------------------------------------------------

def test_each_save_bumps_version_and_invalidates_bills(temp_db):
    _seed()
    main.run_verify()
    _bills, verified_at, v0 = _state()
    assert verified_at is not None

    saves = [
        lambda: main.put_stations(TextIn(text=STATIONS)),
        lambda: main.put_fares(TextIn(text=FARES_2)),
        lambda: main.put_records(TextIn(text=RECORDS_OK)),
        lambda: main.put_config(ConfigIn(
            transfer_minutes=90, transfer_discount=1.0,
            daily_cap=10.0, day_split_hour=4)),
    ]
    for i, save in enumerate(saves, start=1):
        main.run_verify()  # 先造出有效账单
        _b, v_at, _ = _state()
        assert v_at is not None
        save()
        bills, verified_at, version = _state()
        assert version == v0 + i          # 每次保存恰好递增一次
        assert bills == []                # 账单同事务失效
        assert verified_at is None        # 核验时间同事务清除


# ---------------------------------------------------------------------------
# 无冲突：保留原有成功 / 422 行为
# ---------------------------------------------------------------------------

def test_verify_success_without_conflict(temp_db):
    _seed()
    resp = main.run_verify()
    assert not isinstance(resp, JSONResponse)
    assert resp["ok"] is True
    assert resp["bill_count"] == 1
    bills, verified_at, v0 = _state()
    assert len(bills) == 1
    assert verified_at is not None

    # 核验本身不修改输入数据，不递增数据版本
    main.run_verify()
    _b, _v, v1 = _state()
    assert v1 == v0


def test_verify_failure_without_conflict_clears_bills(temp_db):
    _seed()
    main.run_verify()
    assert _state()[0]  # 已有账单

    main.put_records(TextIn(text=RECORDS_BAD))  # 保存坏数据（版本递增）
    resp = main.run_verify()                    # 版本一致：正常失败路径
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 422
    payload = _payload(resp)
    assert payload["ok"] is False
    assert payload["errors"][0]["code"] == "MISSING_EXIT"

    bills, verified_at, _v = _state()
    assert bills == []
    assert verified_at is None


# ---------------------------------------------------------------------------
# 旧数据库迁移：启动时自动补齐数据版本
# ---------------------------------------------------------------------------

def test_init_db_backfills_data_version(temp_db):
    # 模拟旧库：没有 data_version 键
    with db.connect() as conn:
        conn.execute("DELETE FROM meta WHERE key=?", (db.DATA_VERSION_KEY,))
    db.init_db()  # 启动初始化：自动补齐
    with db.connect() as conn:
        assert db.get_data_version(conn) == 0

    # 已有版本保留，不被重置
    with db.connect() as conn:
        db.set_meta(conn, db.DATA_VERSION_KEY, "7")
    db.init_db()
    with db.connect() as conn:
        assert db.get_data_version(conn) == 7
