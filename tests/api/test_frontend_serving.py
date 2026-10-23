"""The API serves the built web app on the same origin (production and Docker)."""

from fastapi.testclient import TestClient

from legible.api.main import create_app


def test_serves_the_single_page_app_safely(settings, tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><title>Legible</title>")
    (dist / "assets" / "app.js").write_text("console.log('app')")
    (dist / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("outside the build")
    settings.serve_frontend = True
    settings.frontend_dist = dist

    with TestClient(create_app(settings)) as client:
        for route in ("/", "/reading", "/talk/deep/link"):
            page = client.get(route)
            assert page.status_code == 200 and "<title>Legible</title>" in page.text
        assert client.get("/assets/app.js").text == "console.log('app')"
        assert client.get("/favicon.svg").headers["content-type"].startswith("image/svg")
        missing_api = client.get("/api/does-not-exist")
        assert missing_api.status_code == 404 and missing_api.json()["detail"]
        for attack in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt"):
            response = client.get(attack)
            assert "outside the build" not in response.text
        assert client.get("/api/health").headers["x-frame-options"] == "DENY"
