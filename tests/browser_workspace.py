"""Optional real-browser regression: pip install playwright; python tests/browser_workspace.py.

Uses installed Google Chrome; starts and stops a local Streamlit server.
Screenshots are written to /tmp/gridlock-preview-*.png.
"""
from pathlib import Path
import math
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

import pandas as pd
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
with socket.socket() as reserved_port:
    reserved_port.bind(('127.0.0.1', 0))
    PORT = reserved_port.getsockname()[1]


def world_point(lat, lon):
    size = 512 * 2**11  # Operations map's initial zoom, flat view.
    return ((lon + 180) / 360 * size,
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * size)


def click_cell(page, row):
    canvas = page.locator('#deckgl-overlay')
    canvas.scroll_into_view_if_needed()
    page.wait_for_function("(() => { const c = document.querySelector('#deckgl-overlay'); return c && Math.abs(c.width / window.devicePixelRatio - c.getBoundingClientRect().width) < 2; })()")
    page.wait_for_timeout(350)  # Allow the resized WebGL viewport to draw a frame.
    box = canvas.bounding_box()
    center = world_point(12.9716, 77.5946)
    target = world_point(row.lat_center, row.lon_center)
    x = box['x'] + box['width']/2 + target[0] - center[0]
    y = box['y'] + box['height']/2 + target[1] - center[1]
    # Wait for an actual pickable hexagon, rather than clicking an uninitialized canvas.
    for attempt in range(12):
        page.mouse.move(x + (attempt % 2), y)
        try:
            page.locator('.deck-tooltip').filter(has_text=str(row.junction_name)).wait_for(timeout=1000)
            break
        except Exception:
            pass
    else:
        raise AssertionError('Expected map-cell tooltip did not render.')
    page.wait_for_timeout(500)
    page.mouse.click(x, y, delay=100)
    try:
        page.get_by_role('heading', name=str(row.junction_name), exact=True).wait_for(timeout=30000)
    except Exception:
        page.screenshot(path='/tmp/gridlock-browser-failure.png')
        raise


def run():
    chrome = shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:
        raise RuntimeError('Install Chrome or Chromium to run this optional browser check.')
    server = subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.headless=true', f'--server.port={PORT}', '--browser.gatherUsageStats=false'],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):
            if server.poll() is not None:
                raise RuntimeError('Streamlit exited before the browser check started.')
            try:
                urllib.request.urlopen(f'http://localhost:{PORT}/_stcore/health', timeout=1)
                break
            except OSError:
                time.sleep(.25)
        else:
            raise RuntimeError('Streamlit did not start within 30 seconds.')
        meta = pd.read_parquet(ROOT / 'cell_metadata.parquet')
        nearest = ((meta.lat_center - 12.9716)**2 + (meta.lon_center - 77.5946)**2).idxmin()
        row = meta.loc[nearest]
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=chrome, headless=True,
                args=['--no-sandbox', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1100})
            errors = []
            page.on('pageerror', lambda error: errors.append(str(error)))
            page.goto(f'http://localhost:{PORT}')
            page.get_by_role('heading', name='Recorded violations by hour', exact=True).wait_for(timeout=60000)
            page.wait_for_timeout(2000)
            click_cell(page, row)
            page.get_by_role('heading', name=str(row.junction_name), exact=True).scroll_into_view_if_needed()
            page.screenshot(path='/tmp/gridlock-preview-desktop-details.png')
            print('PASS: desktop map selection', flush=True)
            # Native priority buttons expose full names and keyboard selection.
            first = page.get_by_role('button', name='1.', exact=False).first
            first_name = first.inner_text().split('. ', 1)[1]
            first.focus()
            page.keyboard.press('Enter')
            page.get_by_role('heading', name=first_name, exact=True).wait_for(timeout=30000)
            page.get_by_role('button', name='Generate reviewed route', exact=True).click()
            page.get_by_role('link', name='Open reviewed route in Google Maps').wait_for(timeout=30000)
            with page.expect_download() as downloading:
                page.get_by_role('button', name='Download patrol brief (CSV)', exact=True).click()
            download = downloading.value
            export = pd.read_csv(download.path())
            assert len(export) == 8
            assert (export.source == 'Historical Records').all()
            assert (export.selected_hours == 'All hours (IST)').all()
            print('PASS: keyboard selection and reviewed CSV export', flush=True)
            page.set_viewport_size({'width': 390, 'height': 844})
            click_cell(page, row)
            page.get_by_role('heading', name=str(row.junction_name), exact=True).scroll_into_view_if_needed()
            page.screenshot(path='/tmp/gridlock-preview-mobile-details.png')
            assert not page.evaluate('document.documentElement.scrollWidth > window.innerWidth')
            print('PASS: mobile map selection and layout', flush=True)
            page.set_viewport_size({'width': 1440, 'height': 1100})
            page.get_by_test_id('stSidebar').get_by_text('Forecast', exact=True).click()
            page.get_by_role('heading', name='Hourly forecast', exact=True).wait_for(timeout=30000)
            click_cell(page, row)
            page.get_by_test_id('stSidebar').get_by_text('Analytics', exact=True).click()
            page.get_by_text('Predicted Enforcement Plan', exact=True).click()
            page.get_by_role('heading', name='Prepare a patrol brief', exact=True).wait_for(timeout=30000)
            assert not page.get_by_test_id('stException').count()
            assert not errors, errors
            browser.close()
        print('PASS: desktop/mobile map clicks, keyboard priority selection, CSV export, Forecast, and Analytics.')
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == '__main__':
    run()
