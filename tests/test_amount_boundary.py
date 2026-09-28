"""金额边界回归测试：十进制金额解析与整数分口径。

覆盖缺陷：
1. 票价文本录入超大金额原本能通过解析，却在保存时触发 SQLite
   整数溢出；现在解析阶段即按十进制金额拒绝（非负、最多两位小数、
   换算成分后处于 SQLite 有符号 64 位范围内），返回带行号的 422；
2. 规则配置（换乘抵扣额 / 每日封顶额）设为超大值后，核验会在写
   账单时失败；现在配置保存阶段即拒绝并返回带字段名的 422；
3. 核验计算与账单落库共用同一整数分口径：处于 64 位边界的票价 /
   封顶额可正常核验并完整落库，账单金额与入库整数分完全一致；
4. 任何非法输入都不改写原配置、票价、账单与数据版本。

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
FARES_OK = "1, 1, 2.00\n1, 2, 3.00\n2, 1, 3.00\n2, 2, 2.00\n"
RECORDS_OK = (
    "A001, 2026-09-18 08:00:00, 科技园, 进\n"
    "A001, 2026-09-18 08:25:00, 中心公园, 出\n"
)

# SQLite 有符号 64 位整数上限（分）与对应元金额
INT64_MAX = core.SQLITE_INT64_MAX
MAX_YUAN_TEXT = core._yuan_text(INT64_MAX)   # 92233720368547758.07
OVER_ONE_CENT_TEXT = "92233720368547758.08"  # 恰好多 1 分
# JSON 侧用 float 传输时可逐分精确承载的大额封顶（2**53 元 = 900719925474099200 分，
# 远在 64 位范围内，且 float -> 十进制文本 -> 分不丢精度）
CAP_YUAN = float(2 ** 53)
CAP_CENTS = 2 ** 53 * 100


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = str(tmp_path / "bus-test.db")
    monkeypatch.setattr(db, "DB_PATH", path)
    db.init_db()
    return path


def _seed(fares: str = FARES_OK, records: str = RECORDS_OK) -> None:
    main.put_stations(TextIn(text=STATIONS))
    main.put_fares(TextIn(text=fares))
    main.put_records(TextIn(text=records))


def _payload(resp: JSONResponse) -> dict:
    return json.loads(resp.body.decode("utf-8"))


def _state():
    with db.connect() as conn:
        return (db.list_bills(conn),
                db.get_meta(conn, "last_verified_at"),
                db.get_data_version(conn),
                db.get_config(conn),
                db.list_fares(conn))


# ---------------------------------------------------------------------------
# 票价文本：十进制金额边界
# ---------------------------------------------------------------------------

def test_parse_amount_cents_boundary():
    assert core.parse_amount_cents("0", "x") == 0
    assert core.parse_amount_cents("3.00", "x") == 300
    assert core.parse_amount_cents("3.5", "x") == 350
    assert core.parse_amount_cents(MAX_YUAN_TEXT, "x") == INT64_MAX
    # 恰好多 1 分即越界
    with pytest.raises(ValueError):
        core.parse_amount_cents(OVER_ONE_CENT_TEXT, "x")
    # 非负 / 两位小数 / 有限数字
    for bad in ("-0.01", "1.234", "0.001", "abc", "nan", "inf",
                "1e999", "9" * 30, "", "  "):
        with pytest.raises(ValueError):
            core.parse_amount_cents(bad, "x")
    # bool 不被当作数字
    with pytest.raises(ValueError):
        core.parse_amount_cents(True, "x")


def test_huge_fare_rejected_with_line_and_unchanged(temp_db):
    _seed()
    main.run_verify()
    bills, verified_at, v0, cfg_before, fares_before = _state()
    assert bills and verified_at is not None

    text = ("1, 1, 2.00\n"
            f"1, 2, {OVER_ONE_CENT_TEXT}\n"   # 第 2 行：超出 64 位
            "2, 1, 1.234\n"                   # 第 3 行：超过两位小数
            "2, 2, -1.00\n")                  # 第 4 行：负数
    resp = main.put_fares(TextIn(text=text))

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 422
    payload = _payload(resp)
    assert payload["ok"] is False
    lines = {e["location"]["line"] for e in payload["errors"]}
    assert lines == {2, 3, 4}
    assert all(e["code"] == "FARE_BAD_VALUE" for e in payload["errors"])

    # 原票价 / 账单 / 核验时间 / 数据版本均保持不变
    bills2, verified_at2, v1, cfg2, fares2 = _state()
    assert v1 == v0
    assert verified_at2 == verified_at
    assert [f["price"] for f in fares2] == [f["price"] for f in fares_before]
    assert bills2 == bills


def test_max_boundary_fare_accepted_and_persisted(temp_db):
    """处于 64 位上界的票价可正常保存、核验并落库（整数分不溢出）。"""
    # 全矩阵四格都给边界票价，确保 1->2 行程确实命中上界票价
    boundary = MAX_YUAN_TEXT
    matrix = (f"1, 1, {boundary}\n1, 2, {boundary}\n"
              f"2, 1, 1.00\n2, 2, {boundary}\n")
    _seed(fares=matrix)
    # 封顶取上界量级（>= 票价），不截断票价
    resp_cfg = main.put_config(ConfigIn(
        transfer_minutes=90, transfer_discount=0,
        daily_cap=float(CAP_YUAN), day_split_hour=4))
    assert not isinstance(resp_cfg, JSONResponse)

    resp = main.run_verify()
    assert not isinstance(resp, JSONResponse)
    bill = resp["bills"][0]
    assert bill["base_cents"] == INT64_MAX
    assert bill["charged_cents"] == CAP_CENTS  # 受封顶约束：min(票价, 封顶)

    # 直接核对入库列：核验整数分与 SQLite 整数列逐分一致（无溢出 / 失真）
    with db.connect() as conn:
        row = conn.execute(
            "SELECT base_cents, charge_cents, cumulative_cents, cap_cents "
            "FROM trip_bill").fetchone()
    assert tuple(row) == (INT64_MAX, CAP_CENTS, CAP_CENTS, CAP_CENTS)

    # 读模型同时提供整数分（权威）与元（展示）两种字段
    with db.connect() as conn:
        listed = db.list_bills(conn)[0]
    assert listed["charged_cents"] == CAP_CENTS
    assert isinstance(listed["charged_fare"], float)


def test_int64_max_fare_without_cap_truncation(temp_db):
    """票价与封顶都取 64 位上界时，实扣逐分等于上界且正常落库。"""
    boundary = MAX_YUAN_TEXT
    matrix = (f"1, 1, {boundary}\n1, 2, {boundary}\n"
              f"2, 1, 1.00\n2, 2, {boundary}\n")
    _seed(fares=matrix)
    # 封顶用文本级精度无法经 float 传输；此处直接以核心 Config 验证计算口径
    with db.connect() as conn:
        db.update_config(conn, core.Config(
            transfer_minutes=90, transfer_discount_cents=0,
            daily_cap_cents=INT64_MAX, day_split_hour=4))
        db.bump_data_version(conn)
        db.clear_bills(conn)
    resp = main.run_verify()
    assert not isinstance(resp, JSONResponse)
    bill = resp["bills"][0]
    assert bill["base_cents"] == INT64_MAX
    assert bill["charged_cents"] == INT64_MAX
    with db.connect() as conn:
        row = conn.execute(
            "SELECT base_cents, charge_cents, cumulative_cents, cap_cents "
            "FROM trip_bill").fetchone()
    assert tuple(row) == (INT64_MAX, INT64_MAX, INT64_MAX, INT64_MAX)


def test_max_boundary_fare_via_http(temp_db):
    """端到端：边界票价经 HTTP 保存、核验、回读一致。"""
    client = TestClient(main.app)
    boundary = MAX_YUAN_TEXT
    client.put("/api/stations", json={"text": STATIONS})
    matrix = (f"1, 1, {boundary}\n1, 2, {boundary}\n"
              f"2, 1, 1.00\n2, 2, {boundary}\n")
    r = client.put("/api/fares", json={"text": matrix})
    assert r.status_code == 200, r.text
    client.put("/api/records", json={"text": RECORDS_OK})
    client.put("/api/config", json={
        "transfer_minutes": 90, "transfer_discount": 0,
        "daily_cap": float(CAP_YUAN), "day_split_hour": 4})
    v = client.post("/api/verify")
    assert v.status_code == 200, v.text
    assert v.json()["bills"][0]["charged_cents"] == CAP_CENTS
    got = client.get("/api/bills").json()
    assert got["items"][0]["charged_cents"] == CAP_CENTS


# ---------------------------------------------------------------------------
# 规则配置：换乘抵扣额 / 每日封顶额
# ---------------------------------------------------------------------------

def test_huge_config_rejected_with_param_and_unchanged(temp_db):
    _seed()
    main.run_verify()
    bills, verified_at, v0, cfg_before, _fares = _state()

    resp = main.put_config(ConfigIn(
        transfer_minutes=90,
        transfer_discount=float(10 ** 30),     # 换算后远超 64 位
        daily_cap=float(10 ** 30),
        day_split_hour=4))

    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 422
    payload = _payload(resp)
    params = {e["location"]["param"] for e in payload["errors"]}
    assert params == {"transfer_discount", "daily_cap"}
    assert all(e["code"] == "CONFIG_BAD" for e in payload["errors"])

    # 配置 / 账单 / 核验时间 / 数据版本保持不变
    bills2, verified_at2, v1, cfg2, _f = _state()
    assert v1 == v0
    assert verified_at2 == verified_at
    assert bills2 == bills
    assert cfg2.transfer_discount_cents == cfg_before.transfer_discount_cents
    assert cfg2.daily_cap_cents == cfg_before.daily_cap_cents


@pytest.mark.parametrize("discount,cap,minutes,hour", [
    (-1, 10, 90, 4),          # 负数抵扣
    (1, -1, 90, 4),           # 负数封顶
    (1, 10, 0, 4),            # 间隔非正
    (1, 10, 1.5, 4),          # 间隔非整数
    (1, 10, 90, 24),          # 分界越界
    (1.234, 10, 90, 4),       # 抵扣超过两位小数
])
def test_invalid_config_variants_rejected(temp_db, discount, cap, minutes, hour):
    resp = main.put_config(ConfigIn(
        transfer_minutes=minutes, transfer_discount=discount,
        daily_cap=cap, day_split_hour=hour))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 422
    assert _payload(resp)["errors"][0]["code"] == "CONFIG_BAD"


def test_huge_daily_cap_does_not_break_verify(temp_db):
    """超大封顶在保存时即被拒绝，核验永远拿不到会溢出的配置。

    边界内的超大封顶（64 位上界）则可正常核验，写账单不失败。
    """
    _seed()
    # 非法封顶被拒绝 -> 配置维持默认 10 元
    rejected = main.put_config(ConfigIn(
        transfer_minutes=90, transfer_discount=1.0,
        daily_cap=10 ** 30, day_split_hour=4))
    assert isinstance(rejected, JSONResponse) and rejected.status_code == 422

    # 合法的上界封顶（元，整数金额，float 可精确承载到该量级）
    ok = main.put_config(ConfigIn(
        transfer_minutes=90, transfer_discount=1.0,
        daily_cap=float(CAP_YUAN), day_split_hour=4))
    assert not isinstance(ok, JSONResponse)
    resp = main.run_verify()
    assert not isinstance(resp, JSONResponse)
    with db.connect() as conn:
        bills = db.list_bills(conn)
    assert bills[0]["charged_cents"] >= 0
    assert bills[0]["cap_cents"] == CAP_CENTS


# ---------------------------------------------------------------------------
# 规则试算对比：草案金额同样按十进制整数分校验
# ---------------------------------------------------------------------------

def test_trial_draft_huge_amount_rejected(temp_db):
    _seed()
    with db.connect() as conn:
        version_before = db.get_data_version(conn)
    client = TestClient(main.app)
    r = client.post("/api/trial-compare", json={
        "transfer_minutes": 90,
        "transfer_discount": 1.0,
        "daily_cap": 10 ** 30,
        "day_split_hour": 4})
    assert r.status_code == 422
    body = r.json()
    assert body["draft"]["ok"] is False
    assert body["draft"]["errors"][0]["code"] == "CONFIG_BAD"
    assert body["draft"]["errors"][0]["location"]["param"] == "daily_cap"
    # 试算只读：配置与数据版本不变（未发生任何保存）
    _bills, _verified_at, version, cfg, _fares = _state()
    assert cfg.daily_cap_cents == 1000
    assert version == version_before
