"""Exercise the default and drift streams in an installed Chrome browser."""

import os
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
            executable = os.environ.get("NUMCOMPUTE_CHROMIUM")
            browser = playwright.chromium.launch(
                executable_path=executable, channel=None if executable else "chrome",
            )
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.goto(address)
            expect(page.get_by_role("heading", name="NumCompute")).to_be_visible()
            expect(page.locator("#kpis")).to_contain_text("94.4%")
            expect(page.locator("#kpis")).to_contain_text("302")
            expect(page.locator(".cover img")).to_have_js_property("complete", True)
            assert page.locator(".cover img").evaluate("(image) => image.naturalWidth > 0")
            page.get_by_label("Batch to inspect").select_option("0")
            expect(page.locator("#chunk-note")).to_contain_text("40 observations used to initialise")
            page.get_by_label("Batch to inspect").select_option("1")
            expect(page.locator("#chunk-note")).to_contain_text("scored before updating")
            page.locator("#diagnostics summary").click()
            expect(page.locator('#rows tr[data-selected="true"]')).to_contain_text("2")
            page.get_by_label("Dataset").select_option("drift")
            page.get_by_label("Observations per batch").select_option("80")
            page.get_by_role("button", name="Run evaluation").click()
            expect(page.locator("#source")).to_contain_text("reverses after sample 160")
            expect(page.locator("#kpis")).to_contain_text("240")
            expect(page.locator("#rows tr")).to_have_count(4)
            page.route("**/api/experiment?*", lambda route: route.fulfill(
                status=503, json={"error": "Experiment temporarily unavailable."},
            ))
            page.get_by_role("button", name="Run evaluation").click()
            expect(page.get_by_role("status")).to_contain_text("previous run")
            expect(page.locator("#kpis")).to_contain_text("240")
            assert not errors, errors

            previews = os.environ.get("NUMCOMPUTE_PREVIEWS")
            desktop = browser.new_page(viewport={"width": 1440, "height": 1000})
            desktop.goto(address)
            expect(desktop.locator("#kpis")).to_contain_text("94.4%")
            desktop.keyboard.press("Tab")
            expect(desktop.locator(":focus-visible")).to_have_count(1)
            desktop.locator(":focus-visible").evaluate("(element) => element.blur()")
            mobile = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
            mobile.goto(address)
            expect(mobile.locator("#kpis")).to_contain_text("94.4%")
            assert mobile.evaluate("document.documentElement.scrollWidth === innerWidth")
            if previews:
                folder = Path(previews)
                folder.mkdir(parents=True, exist_ok=True)
                desktop.screenshot(path=folder / "numcompute-lab-desktop.png", full_page=True)
                mobile.screenshot(path=folder / "numcompute-lab-mobile.png", full_page=True)
    finally:
        server.terminate()
