"""Screenshots of the built interface for the README and docs, taken with headless Chromium.

Requires playwright and a Chromium build (`python -m playwright install chromium`).
"""
from __future__ import annotations

from pathlib import Path

from .. import config

OUT = config.ROOT / "docs" / "screenshots"


def run() -> list[str]:
    from playwright.sync_api import sync_playwright

    OUT.mkdir(parents=True, exist_ok=True)
    page_url = (config.WEB / "dist" / "index.html").resolve().as_uri()
    shots = []
    with sync_playwright() as p:
        browser = p.chromium.launch()

        def open_page(width=1440, height=1000, dark=False, scale=1.25):
            pg = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=scale,
                                  color_scheme="dark" if dark else "light")
            pg.goto(page_url)
            # Full page captures stitch the page; a sticky header would land mid image.
            pg.add_style_tag(content="header.top{position:static !important}")
            pg.wait_for_function("!document.querySelector('#src-live').disabled", timeout=120000)
            return pg

        def snap(pg, name, full=True, selector=None):
            path = OUT / name
            if selector:
                pg.locator(selector).screenshot(path=str(path))
            else:
                pg.screenshot(path=str(path), full_page=full)
            shots.append(name)

        pg = open_page()
        pg.click("#tab-case")
        pg.select_option("#case-pick", "0")
        pg.locator("#reveal summary").click()
        pg.wait_for_timeout(400)
        snap(pg, "case.png")
        snap(pg, "evidence.png", selector="#cand")
        snap(pg, "summary.png", selector=".col-c .panel:nth-of-type(2)")
        pg.click("#src-live")
        pg.wait_for_timeout(1500)
        snap(pg, "note.png")
        pg.locator("#questions .btn").first.click()
        pg.wait_for_timeout(1200)
        snap(pg, "question_answered.png", selector=".col-b")
        for view in ("results", "leakage", "safety", "knowledge", "method"):
            pg.click(f"#tab-{view}")
            pg.wait_for_timeout(300)
            if view == "knowledge":
                pg.fill("#kb-q", "Noonan syndrome 1")
                pg.wait_for_timeout(500)
            snap(pg, f"{view}.png")
        pg.close()

        pg = open_page(dark=True)
        pg.click("#tab-case")
        snap(pg, "dark.png", full=False)
        pg.close()

        pg = open_page(width=400, height=860, scale=2)
        pg.click("#tab-case")
        snap(pg, "phone.png", full=False)
        pg.close()
        browser.close()
    return shots


if __name__ == "__main__":
    print(run())
