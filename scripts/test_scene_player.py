"""Browser integration check. Usage: python test_scene_player.py path/to/index.html screenshot.png"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright

def check(html, screenshot):
    errors=[];requests=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'],timeout=20000)
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda request:requests.append(request.url))
        page.goto(html.resolve().as_uri())
        page.wait_for_function('current !== null && current.index === 0')
        count=page.evaluate('scene.frames.length')
        assert count>1
        page.locator('#next').click()
        page.wait_for_function('current.index === 1')
        page.locator('#prev').click()
        page.wait_for_function('current.index === 0')
        page.locator('#showTruth').check()
        page.locator('#showPoints').uncheck()
        page.locator('#showPoints').check()
        if page.locator('#detections button').count():
            page.locator('#detections button').first.click()
            assert page.locator('#detections button.selected').count()==1
        middle=count//2
        page.evaluate('(i)=>{const t=document.getElementById("timeline");t.value=scene.frames[i].t_ms;t.dispatchEvent(new Event("input"));}',middle)
        page.wait_for_function('(i)=>current.index===i',arg=middle)
        page.locator('#wheel').hover();page.mouse.wheel(0,80)
        page.wait_for_function('(i)=>current.index===i',arg=min(middle+1,count-1))
        page.evaluate('seek(0)');page.wait_for_function('current.index===0')
        page.locator('#speed').select_option('2')
        page.locator('#play').click()
        page.wait_for_function('current.index>0')
        page.locator('#play').click()
        assert page.locator('#play').inner_text()=='Play'
        # Seek backward and forward rapidly to test asynchronous frame loading.
        page.evaluate('seek(duration);seek(0);seek(scene.frames[Math.floor(scene.frames.length/3)].t_ms)')
        page.wait_for_function('current.index===Math.floor(scene.frames.length/3)')
        page.screenshot(path=str(screenshot),full_page=True)
        assert not errors,errors
        assert all(url.startswith(('file:','data:')) for url in requests),requests
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=window.innerWidth'), 'mobile horizontal overflow'
        browser.close()
        print(f'Passed: local-file load, {count} frames, stepping, overlays, selection, timeline, wheel, playback, rapid seek, mobile width, no external requests.')

if __name__=='__main__':check(Path(sys.argv[1]),Path(sys.argv[2]))
