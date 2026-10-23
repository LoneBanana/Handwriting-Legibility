import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from legible.api.main import create_app
from legible.api.services.clarify import Clarified
from legible.htr.checkpoint import inference_payload, save_atomic

from ..conftest import register
from ..htr.helpers import synthetic_page, tiny_model


def _png(gray: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(gray).save(buffer, format="PNG")
    return buffer.getvalue()


PAGE = _png(synthetic_page(["first line here", "second line", "third"], seed=1))


@pytest.fixture
def model_client(settings, tmp_path):
    model, vocab = tiny_model(seed=7)
    path = tmp_path / "tiny.pt"
    save_atomic(inference_payload(model.cfg, vocab, model.state_dict(), {"version": "tiny-test"}), path)
    settings.htr_checkpoint = path
    settings.htr_device = "cpu"
    settings.htr_beam_size = 2
    with TestClient(create_app(settings)) as client:
        yield client


def _headers(client):
    return {"Authorization": f"Bearer {register(client)['access_token']}", "X-Timezone": "UTC"}


def _post(client, headers, data=PAGE, content_type="image/png", **fields):
    form = {"mode": "page", "decoder": "fast", **{k: str(v).lower() for k, v in fields.items()}}
    return client.post("/api/handwriting/transcribe", headers=headers,
                       files={"file": ("page.png", data, content_type)}, data=form)


def test_without_a_model_the_api_explains_how_to_set_one_up(client, auth_headers):
    response = _post(client, auth_headers)
    assert response.status_code == 503 and "LEGIBLE_HTR_CHECKPOINT" in response.json()["detail"]
    status = client.get("/api/handwriting/status", headers=auth_headers).json()
    assert status["model_loaded"] is False


def test_rejects_wrong_types_and_oversized_files(client, auth_headers, app):
    assert _post(client, auth_headers, content_type="application/pdf").status_code == 415
    app.state.settings.max_upload_bytes = 10
    assert _post(client, auth_headers).status_code == 413


def test_transcribes_and_feeds_the_clarity_track(model_client):
    headers = _headers(model_client)
    response = _post(model_client, headers)
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["lines"]) == 3 and body["model_version"] == "tiny-test"
    assert body["sample_id"] is None and body["clarified"] is None and body["personalized"] is False
    assert 0.0 <= body["legibility"] <= 1.0
    (update,) = body["progressions"]
    assert update["track"] == "handwriting.legibility"
    assert _post(model_client, headers, track_progress=False).json()["progressions"] == []
    assert model_client.get("/api/handwriting/status", headers=headers).json()["model_loaded"] is True
    bad = _post(model_client, headers, data=b"not an image")
    assert bad.status_code == 422


def test_consented_samples_can_be_corrected_and_deleted(model_client):
    headers = _headers(model_client)
    model_client.patch("/api/auth/me/settings", json={"store_handwriting_samples": True}, headers=headers)
    body = _post(model_client, headers).json()
    sample_id = body["sample_id"]
    assert sample_id
    settings = model_client.app.state.settings
    user_id = model_client.get("/api/auth/me", headers=headers).json()["id"]
    folder = settings.uploads_dir / user_id
    assert (folder / f"{sample_id}.png").exists() and (folder / f"{sample_id}.npz").exists()

    fixed = model_client.post(f"/api/handwriting/samples/{sample_id}/correction",
                              json={"text": "first line here\nsecond line\nthird"}, headers=headers)
    assert fixed.status_code == 200 and fixed.json() == {"ok": True}
    assert model_client.get("/api/handwriting/samples/count", headers=headers).json() == {"samples": 1, "corrected": 1}

    other = _headers(model_client)
    assert model_client.post(f"/api/handwriting/samples/{sample_id}/correction", json={"text": "x"},
                             headers=other).status_code == 404

    assert model_client.delete("/api/handwriting/samples", headers=headers).status_code == 204
    assert not folder.exists()
    assert model_client.get("/api/handwriting/samples/count", headers=headers).json()["samples"] == 0


class FakeLLM:
    available = True

    def __init__(self):
        self.prompts = []

    async def structured(self, *, system, messages, schema, max_tokens=4096, effort="medium"):
        self.prompts.append(messages[0]["content"])
        return Clarified(text="First line here.", changes=[{"original": "frist", "corrected": "First",
                                                            "reason": "spelling"}], uncertain=["third"])

    async def aclose(self):
        return None


def test_clarify_uses_claude_with_uncertain_letters_marked(model_client):
    fake = FakeLLM()
    model_client.app.state.llm = fake
    body = _post(model_client, _headers(model_client), clarify=True).json()
    assert body["clarified"] == {"text": "First line here.",
                                 "changes": [{"original": "frist", "corrected": "First", "reason": "spelling"}],
                                 "uncertain": ["third"]}
    assert fake.prompts and fake.prompts[0].startswith("Transcribed lines:")


def test_enough_corrections_train_a_personal_adapter(model_client):
    from sqlalchemy import select

    from legible.api.models import HandwritingSample

    app = model_client.app
    app.state.settings.personalize_min_samples = 6
    app.state.settings.personalize_steps = 4
    headers = _headers(model_client)
    model_client.patch("/api/auth/me/settings", json={"store_handwriting_samples": True}, headers=headers)
    worker = app.state.personalizer
    for _ in range(2):
        sample_id = _post(model_client, headers).json()["sample_id"]
        model_client.post(f"/api/handwriting/samples/{sample_id}/correction",
                          json={"text": "first line here\nsecond line\nthird"}, headers=headers)
    worker._executor.submit(lambda: None).result(timeout=300)  # wait for the queued job
    with app.state.sessionmaker() as db:
        samples = db.scalars(select(HandwritingSample)).all()
    assert len(samples) == 2 and all(s.used_in_adapter for s in samples)
    user_id = model_client.get("/api/auth/me", headers=headers).json()["id"]
    adapter = app.state.handwriting.adapter_path(user_id)
    status = model_client.get("/api/handwriting/status", headers=headers).json()
    assert status["personalized"] == adapter.exists()  # kept only if it beat the shared model
