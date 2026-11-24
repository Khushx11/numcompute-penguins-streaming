"""Exercise the default and drift streams in an installed Chrome browser."""

from pathlib import Path
import subprocess
import sys

from playwright.sync_api import expect, sync_playwright


with subprocess.Popen(
    [sys.executable, str(Path(__file__).resolve().parents[1] / "portfolio_demo.py"),
     "--port", "0"],
    stdout=subprocess.PIPE, text=True,
) as server:
    try:
        address = server.stdout.readline().strip().removeprefix("Open ")
        assert address.startswith("http://127.0.0.1:"), address
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chrome")
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(address)
            expect(page.get_by_role("heading", name="Streaming evaluation")).to_be_visible()
            expect(page.locator("#kpis")).to_contain_text("94.4%")
            expect(page.locator("#kpis")).to_contain_text("302")
            page.get_by_label("Dataset").select_option("drift")
            page.get_by_label("Chunk size").select_option("80")
            page.get_by_role("button", name="Run evaluation").click()
            expect(page.locator("#source")).to_contain_text("reverses after sample 160")
            expect(page.locator("#kpis")).to_contain_text("240")
            expect(page.locator("#rows tr")).to_have_count(4)
            assert not errors, errors
    finally:
        server.terminate()
