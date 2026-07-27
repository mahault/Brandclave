"""Tests for auth routes, saved-items CRUD, and per-user blueprint filtering.

Uses a temporary SQLite database: DATABASE_URL is set *before* importing the
app so db.database binds its engine to the temp file. Scheduler and service
pre-warming are disabled to keep the tests fast and self-contained.
"""

import os
import tempfile
import uuid

# --- Environment must be configured BEFORE importing the app ---
_db_fd, _db_path = tempfile.mkstemp(prefix="brandclave_test_", suffix=".db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = "sqlite:///" + _db_path.replace("\\", "/")
os.environ["SCHEDULER_ENABLED"] = "false"
os.environ["PREWARM_SERVICES"] = "false"
os.environ["JWT_SECRET"] = "test-secret-not-for-production-0123456789abcdef"

from fastapi.testclient import TestClient

from api.main import app
from db.database import get_db_session, init_db
from db.models import BrandBlueprintModel

init_db()

client = TestClient(app)


def _unique_email(prefix: str = "user") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}@example.com"


def _register(email: str, password: str = "hunter2secure", display_name: str | None = None):
    payload = {"email": email, "password": password}
    if display_name:
        payload["display_name"] = display_name
    return client.post("/api/auth/register", json=payload)


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# --- Register / login / me ---


def test_register_login_me_happy_path():
    email = _unique_email()
    resp = _register(email, "hunter2secure", display_name="Test User")
    assert resp.status_code == 201
    body = resp.json()
    assert body["token"]
    assert body["user"]["email"] == email
    assert body["user"]["display_name"] == "Test User"
    assert "password" not in body["user"]
    assert "password_hash" not in body["user"]

    login = client.post(
        "/api/auth/login", json={"email": email, "password": "hunter2secure"}
    )
    assert login.status_code == 200
    token = login.json()["token"]

    me = client.get("/api/auth/me", headers=_auth_headers(token))
    assert me.status_code == 200
    assert me.json()["email"] == email


def test_register_normalizes_email_case():
    email = _unique_email()
    resp = _register(email.upper())
    assert resp.status_code == 201
    assert resp.json()["user"]["email"] == email.lower()

    # Login with a different casing still works
    login = client.post(
        "/api/auth/login", json={"email": email.title(), "password": "hunter2secure"}
    )
    assert login.status_code == 200


def test_register_duplicate_email_409():
    email = _unique_email()
    assert _register(email).status_code == 201
    assert _register(email).status_code == 409


def test_register_short_password_422():
    resp = _register(_unique_email(), password="short")
    assert resp.status_code == 422


def test_login_bad_password_401():
    email = _unique_email()
    _register(email)
    resp = client.post("/api/auth/login", json={"email": email, "password": "wrongpassword"})
    assert resp.status_code == 401


def test_login_unknown_email_401():
    resp = client.post(
        "/api/auth/login",
        json={"email": _unique_email("ghost"), "password": "whatever123"},
    )
    assert resp.status_code == 401


def test_me_missing_or_invalid_token_401():
    assert client.get("/api/auth/me").status_code == 401
    assert (
        client.get("/api/auth/me", headers=_auth_headers("not-a-jwt")).status_code == 401
    )


# --- Saved items CRUD ---


def test_saved_items_require_auth_401():
    assert client.get("/api/projects/saved").status_code == 401
    assert (
        client.post(
            "/api/projects/saved",
            json={"item_type": "trend", "item_id": "abc", "title": "T"},
        ).status_code
        == 401
    )
    assert client.delete("/api/projects/saved/some-id").status_code == 401


def test_saved_items_crud():
    token = _register(_unique_email("saver")).json()["token"]
    headers = _auth_headers(token)

    # Empty to start
    resp = client.get("/api/projects/saved", headers=headers)
    assert resp.status_code == 200
    assert resp.json() == {"items": [], "total": 0}

    # Save a trend with a snapshot
    payload = {
        "item_type": "trend",
        "item_id": "trend-123",
        "title": "Wellness sabbaticals",
        "snapshot": {"name": "Wellness sabbaticals", "strength_score": 0.82},
    }
    created = client.post("/api/projects/saved", json=payload, headers=headers)
    assert created.status_code == 201
    saved = created.json()
    assert saved["item_type"] == "trend"
    assert saved["item_id"] == "trend-123"
    assert saved["snapshot"]["strength_score"] == 0.82

    # Duplicate save is rejected
    dup = client.post("/api/projects/saved", json=payload, headers=headers)
    assert dup.status_code == 409

    # Listed for the owner
    listed = client.get("/api/projects/saved", headers=headers)
    assert listed.json()["total"] == 1
    assert listed.json()["items"][0]["id"] == saved["id"]

    # Invisible to another user
    other_token = _register(_unique_email("other")).json()["token"]
    other_list = client.get("/api/projects/saved", headers=_auth_headers(other_token))
    assert other_list.json()["total"] == 0

    # Another user cannot delete it
    other_delete = client.delete(
        f"/api/projects/saved/{saved['id']}", headers=_auth_headers(other_token)
    )
    assert other_delete.status_code == 404

    # Owner deletes it
    deleted = client.delete(f"/api/projects/saved/{saved['id']}", headers=headers)
    assert deleted.status_code == 200
    assert client.get("/api/projects/saved", headers=headers).json()["total"] == 0


def test_saved_item_invalid_type_422():
    token = _register(_unique_email("typed")).json()["token"]
    resp = client.post(
        "/api/projects/saved",
        json={"item_type": "hotel", "item_id": "x"},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 422


# --- Blueprint list filtering by user ---


def _make_blueprint_row(user_id: str | None, location: str) -> str:
    """Insert a minimal BrandBlueprintModel row directly and return its id."""
    session = get_db_session()
    try:
        model = BrandBlueprintModel(
            location=location,
            segment="lifestyle",
            adr=250.0,
            rooms=100,
            developer_goal="A test development goal for filtering",
            brand_name_primary="Testmark",
            one_liner="A test blueprint",
            thesis="Test thesis",
            positioning_statement="Positioned for tests",
            design_direction="Minimal",
            revenue_logic="Rooms plus F&B",
            investor_summary="Solid test returns",
            user_id=user_id,
        )
        session.add(model)
        session.commit()
        return model.id
    finally:
        session.close()


def test_blueprint_list_filters_by_user():
    # Two users plus one anonymous (NULL user_id) row
    location = f"Testville-{uuid.uuid4().hex[:8]}"
    reg_a = _register(_unique_email("owner-a")).json()
    reg_b = _register(_unique_email("owner-b")).json()
    user_a_id = reg_a["user"]["id"]
    user_b_id = reg_b["user"]["id"]

    id_a = _make_blueprint_row(user_a_id, location)
    id_b = _make_blueprint_row(user_b_id, location)
    id_anon = _make_blueprint_row(None, location)

    # Authenticated as A: own row + anonymous row, B's row excluded
    resp = client.get(
        "/api/brand-blueprint",
        params={"location": location},
        headers=_auth_headers(reg_a["token"]),
    )
    assert resp.status_code == 200
    ids = {b["id"] for b in resp.json()["blueprints"]}
    assert ids == {id_a, id_anon}
    assert id_b not in ids

    # Anonymous request keeps the legacy behavior: sees everything
    anon_resp = client.get("/api/brand-blueprint", params={"location": location})
    assert anon_resp.status_code == 200
    anon_ids = {b["id"] for b in anon_resp.json()["blueprints"]}
    assert anon_ids == {id_a, id_b, id_anon}
