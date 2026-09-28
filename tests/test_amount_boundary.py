"""金额边界回归测试：统一十进制金额解析 + 整数分核验 / 落库口径。

覆盖修复的缺陷：
1. 录入超大票价旧实现可通过解析，却在保存账单时触发 SQLite 整数溢出
   （500）；现在解析阶段即按十进制金额口径拒绝（422，带行号），合法上限
   （换算为分恰为 SQLite 有符号 64 位最大值）可完整核验落库；
2. 每日封顶额 / 换乘抵扣额设为超大值后，核验写账单同样溢出；现在配置
   保存阶段即拒绝（422，带字段名 location.param），原配置、票价、账单
   与数据版本均保持不变；
3. 核验计算与账单落库使用同一整数分口径，边界值不经浮点转换，
   不溢出、不失真；
4. 三位小数 / 负数 / 非数字 / 科学计数法等非法金额一律 422；
5. 旧版本以「元」文本保存的配置在升级后仍可读（迁移兼容）。

运行：``python3 -m pytest tests/ -q``。
"""
from __future__ import annotations

import json

import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from app import core, db, main
from app.core import (
    SQLITE_INT64_MAX,
    build_config_from_input,
    parse_amount_cents,
)
from app.schemas import ConfigIn, TextIn, TrialConfigIn

STATIONS = "科技园, 1\n中心公园, 2\n"
RECORDS = (
    "A001, 2026-09-18 08:00:00, 科技园, 进\n"
    "A001, 2026-09-18 08:25:00, 中心公园, 出\n"
)

# 换算为分恰好等于 SQLite 有符号 64 位整数最大值的金额（元）
MAX_YUAN = "92233720368547758.07"
MAX_CENTS = SQLITE_INT64_MAX
OVER_BY_ONE_YUAN = "92233720368547758.08"
HUGE_YUAN = "9" * 30


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "bus-test.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    return path


def _payload(resp: JSONResponse) -> dict:
    return json.loads(resp.body.decode("utf-8"))


def _seed(fares: str, records: str = RECORDS) -> None:
    main.put_stations(TextIn(text=STATIONS))
    main.put_fares(TextIn(text=fares))
    main.put_records(TextIn(text=records))


# ---------------------------------------------------------------------------
# 纯解析层：parse_amount_cents
# ---------------------------------------------------------------------------

def test_parse_amount_cents_accepts_decimal_shapes():
    assert parse_amount_cents("0") == 0
    assert parse_amount_cents("3") == 300
    assert parse_amount_cents("3.1") == 310
    assert parse_amount_cents("3.10") == 310
    assert parse_amount_cents("0.01") == 1
    assert parse_amount_cents(" 12.34 ") == 1234
    assert parse_amount_cents(MAX_YUAN) == MAX_CENTS


@pytest.mark.parametrize("text", [
    "", "   ", "abc", "-1", "-0.01", "1e3", "1.2.3", ".5", "+1",
    "3.123", "0.001", OVER_BY_ONE_YUAN, HUGE_YUAN,
])
def test_parse_amount_cents_rejects_bad_values(text):
    with pytest.raises(ValueError):
        parse_amount_cents(text)


def test_build_config_from_input_locates_bad_fields():
    cfg, errors = build_config_from_input(
        90, 1.0, HUGE_YUAN, 4)
    assert cfg is None
    assert [e["location"]["param"] for e in errors] == ["daily_cap"]

    cfg, errors = build_config_from_input(
        90, 1.005, -1.0, 24)
    params = {e["location"]["param"] for e in errors}
    assert params == {"transfer_discount", "daily_cap", "day_split_hour"}

    cfg, errors = build_config_from_input(
        1.5, 1.0, 10.0, 4)
    assert cfg is None
    assert errors[0]["location"]["param"] == "transfer_minutes"

    cfg, errors = build_config_from_input(
        90, "1.50", MAX_YUAN, 4)
    assert errors is None
    assert cfg.transfer_discount_cents == 150
    assert cfg.daily_cap_cents == MAX_CENTS


# ---------------------------------------------------------------------------
# 票价文本：超大票价解析阶段 422（带行号），不再到保存账单时溢出
# ---------------------------------------------------------------------------

def test_huge_fare_rejected_at_parse_with_line(temp_db):
    resp = main.put_fares(TextIn(text=f"1, 2, {HUGE_YUAN}\n"))
    assert isinstance(resp, JSONResponse) and resp.status_code == 422
    body = _payload(resp)
    err = body["errors"][0]
    assert err["code"] == "FARE_BAD_VALUE"
    assert err["location"]["line"] == 1


def test_three_decimal_fare_rejected_with_line(temp_db):
    resp = main.put_fares(TextIn(text="1, 2, 3.00\n2, 1, 3.123\n"))
    body = _payload(resp)
    assert resp.status_code == 422
    assert body["errors"][0]["code"] == "FARE_BAD_VALUE"
    assert body["errors"][0]["location"]["line"] == 2


def test_boundary_fare_verifies_and_persists_exact_cents(temp_db):
    _seed(f"1, 2, {MAX_YUAN}\n")
    resp = main.run_verify()
    assert not isinstance(resp, JSONResponse)
    assert resp["bill_count"] == 1
    with db.connect() as conn:
        row = conn.execute(
            "SELECT base_cents, charge_cents FROM trip_bill").fetchone()
    # 基础票价精确落为 INT64_MAX 分；默认封顶 10 元，实扣 10 元
    assert row["base_cents"] == MAX_CENTS
    assert row["charge_cents"] == 1000


def test_over_boundary_fare_rejected(temp_db):
    resp = main.put_fares(TextIn(text=f"1, 2, {OVER_BY_ONE_YUAN}\n"))
    assert resp.status_code == 422
    assert _payload(resp)["errors"][0]["code"] == "FARE_BAD_VALUE"


def test_invalid_fare_text_leaves_table_version_bills_unchanged(temp_db):
    _seed("1, 2, 3.00\n2, 1, 3.00\n")
    main.run_verify()
    with db.connect() as conn:
        v0 = db.get_data_version(conn)
        fares_before = conn.execute(
            "SELECT from_zone, to_zone, price_cents FROM fare").fetchall()
        bills_before = conn.execute("SELECT COUNT(*) c FROM trip_bill").fetchone()["c"]

    resp = main.put_fares(TextIn(text="1, 2, 3.00\n2, 1, oops\n"))
    assert resp.status_code == 422

    with db.connect() as conn:
        assert db.get_data_version(conn) == v0
        fares_after = conn.execute(
            "SELECT from_zone, to_zone, price_cents FROM fare").fetchall()
        bills_after = conn.execute("SELECT COUNT(*) c FROM trip_bill").fetchone()["c"]
    assert [tuple(r) for r in fares_after] == [tuple(r) for r in fares_before]
    assert bills_after == bills_before == 1


# ---------------------------------------------------------------------------
# 规则配置：超大抵扣 / 封顶保存即 422（带字段），数据保持不变
# ---------------------------------------------------------------------------

def test_huge_daily_cap_rejected_with_field(temp_db):
    _seed("1, 2, 3.00\n")
    resp = main.put_config(ConfigIn(
        transfer_minutes=90, transfer_discount=1.0,
        daily_cap=HUGE_YUAN, day_split_hour=4))
    assert resp.status_code == 422
    err = _payload(resp)["errors"][0]
    assert err["code"] == "CONFIG_BAD"
    assert err["location"]["param"] == "daily_cap"


def test_huge_transfer_discount_rejected_with_field(temp_db):
    resp = main.put_config(ConfigIn(
        transfer_minutes=90, transfer_discount=1e30,
        daily_cap=10.0, day_split_hour=4))
    assert resp.status_code == 422
    assert _payload(resp)["errors"][0]["location"]["param"] == \
        "transfer_discount"


def test_invalid_config_changes_nothing(temp_db):
    _seed("1, 2, 3.00\n")
    main.run_verify()
    with db.connect() as conn:
        v0 = db.get_data_version(conn)
        cfg0 = db.get_config(conn)
        bills0 = conn.execute("SELECT COUNT(*) c FROM trip_bill").fetchone()["c"]
        verified_at0 = db.get_meta(conn, "last_verified_at")

    resp = main.put_config(ConfigIn(
        transfer_minutes=0, transfer_discount=-1.0,
        daily_cap=1.005, day_split_hour=99))
    assert resp.status_code == 422

    with db.connect() as conn:
        assert db.get_data_version(conn) == v0
        assert db.get_config(conn) == cfg0
        assert conn.execute("SELECT COUNT(*) c FROM trip_bill").fetchone()["c"] == bills0
        assert db.get_meta(conn, "last_verified_at") == verified_at0


def test_boundary_cap_persists_exact_and_verify_succeeds(temp_db):
    _seed("1, 2, 100.00\n")
    resp = main.put_config(ConfigIn(
        transfer_minutes=90, transfer_discount=1.0,
        daily_cap=MAX_YUAN, day_split_hour=4))
    assert not isinstance(resp, JSONResponse)
    verify_resp = main.run_verify()
    assert not isinstance(verify_resp, JSONResponse)
    bill = verify_resp["bills"][0]
    assert bill["charged_fare"] == 100.0
    with db.connect() as conn:
        row = conn.execute(
            "SELECT charge_cents, cap_cents FROM trip_bill").fetchone()
    assert row["charge_cents"] == 10000
    assert row["cap_cents"] == MAX_CENTS


# ---------------------------------------------------------------------------
# 试算草案：超大金额 422，仅 draft 分块报错并定位字段；current 正常
# ---------------------------------------------------------------------------

def test_trial_draft_huge_amount_422_with_field(temp_db):
    _seed("1, 2, 3.00\n")
    resp = main.run_trial_compare(TrialConfigIn(
        transfer_minutes=90, transfer_discount=1e30,
        daily_cap=10.0, day_split_hour=4))
    assert isinstance(resp, JSONResponse) and resp.status_code == 422
    body = _payload(resp)
    assert body["draft"]["ok"] is False
    assert body["draft"]["errors"][0]["location"]["param"] == \
        "transfer_discount"
    assert body["current"]["ok"] is True
    # 试算只读：账单与版本不变
    with db.connect() as conn:
        assert db.get_meta(conn, "last_verified_at") is None


# ---------------------------------------------------------------------------
# 整数分口径下的普通计费回归
# ---------------------------------------------------------------------------

def test_pricing_regression_transfer_and_cap_in_cents(temp_db):
    _seed("1, 2, 3.00\n2, 1, 3.00\n", records=(
        "A001, 2026-09-18 08:00:00, 科技园, 进\n"
        "A001, 2026-09-18 08:20:00, 中心公园, 出\n"
        "A001, 2026-09-18 08:30:00, 中心公园, 进\n"
        "A001, 2026-09-18 08:50:00, 科技园, 出\n"
    ))
    resp = main.run_verify()
    bills = resp["bills"]
    assert [b["charged_fare"] for b in bills] == [3.0, 2.0]
    assert bills[1]["daily_cumulative"] == 5.0


# ---------------------------------------------------------------------------
# HTTP 端到端：错误形态与原配置不变
# ---------------------------------------------------------------------------

def test_http_huge_fare_and_cap_return_422(temp_db):
    client = TestClient(main.app)
    r = client.put("/api/fares", json={"text": f"1, 2, {HUGE_YUAN}\n"})
    assert r.status_code == 422
    assert r.json()["errors"][0]["location"]["line"] == 1

    r = client.put("/api/config", json={
        "transfer_minutes": 90, "transfer_discount": 1.0,
        "daily_cap": OVER_BY_ONE_YUAN, "day_split_hour": 4})
    assert r.status_code == 422
    assert r.json()["errors"][0]["location"]["param"] == "daily_cap"


# ---------------------------------------------------------------------------
# 旧库迁移：旧版本元文本金额键升级后仍生效，首次保存后切换到分键
# ---------------------------------------------------------------------------

def test_legacy_yuan_text_config_still_reads(temp_db):
    with db.connect() as conn:
        db.set_meta(conn, "transfer_discount", "1.50")
        db.set_meta(conn, "daily_cap", "12.30")
        conn.execute(
            "DELETE FROM meta WHERE key IN "
            "('transfer_discount_cents', 'daily_cap_cents')")
    with db.connect() as conn:
        cfg = db.get_config(conn)
    assert cfg.transfer_discount_cents == 150
    assert cfg.daily_cap_cents == 1230

    # 保存一次新配置后，旧键清除、新键生效
    main.put_config(ConfigIn(
        transfer_minutes=90, transfer_discount=1.0,
        daily_cap=10.0, day_split_hour=4))
    with db.connect() as conn:
        keys = {r["key"] for r in conn.execute("SELECT key FROM meta")}
    assert "transfer_discount" not in keys
    assert "daily_cap" not in keys
    assert "transfer_discount_cents" in keys
    assert "daily_cap_cents" in keys
