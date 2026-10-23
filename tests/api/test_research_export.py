import json

from legible.api.research_export import export, pseudonym

from .test_handwriting_api import _headers, _post, model_client  # noqa: F401  (fixture)

CORRECTION = "first line here\nsecond line\nthird"


def _sample_with_correction(client, headers, share: bool) -> str:
    client.patch("/api/auth/me/settings", json={"store_handwriting_samples": True,
                                                "share_samples_for_research": share}, headers=headers)
    sample_id = _post(client, headers).json()["sample_id"]
    client.post(f"/api/handwriting/samples/{sample_id}/correction", json={"text": CORRECTION}, headers=headers)
    return client.get("/api/auth/me", headers=headers).json()["id"]


def test_exports_only_consenting_writers_pseudonymously(model_client, tmp_path):  # noqa: F811
    sharer = _headers(model_client)
    private = _headers(model_client)
    sharer_id = _sample_with_correction(model_client, sharer, share=True)
    _sample_with_correction(model_client, private, share=False)

    settings = model_client.app.state.settings
    stats = export(settings, tmp_path / "export", model_client.app.state.sessionmaker)
    assert stats["writers"] == 1 and stats["lines"] == 3

    rows = [json.loads(line) for line in (tmp_path / "export" / "all.jsonl").read_text().splitlines()]
    writer = pseudonym(sharer_id, settings.secret_key())
    assert {r["writer"] for r in rows} == {writer} and writer != sharer_id
    assert [r["text"] for r in rows] == CORRECTION.split("\n")
    assert all((tmp_path / "export" / r["image"]).exists() for r in rows)
    dump = (tmp_path / "export" / "all.jsonl").read_text()
    assert "@" not in dump and sharer_id not in dump  # no emails or account ids


def test_sharing_turns_off_when_samples_are_not_kept(model_client):  # noqa: F811
    headers = _headers(model_client)
    on = model_client.patch("/api/auth/me/settings", headers=headers,
                            json={"store_handwriting_samples": True, "share_samples_for_research": True}).json()
    assert on["settings"]["share_samples_for_research"] is True
    off = model_client.patch("/api/auth/me/settings", headers=headers, json={"store_handwriting_samples": False}).json()
    assert off["settings"]["share_samples_for_research"] is False
