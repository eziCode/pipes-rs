// Exercise the shipped player without a browser or third-party dependencies.
const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const source = fs.readFileSync('assets/scene.html','utf8').split('<script>')[1].split('</script>')[0];
function player(timestamps=[1000000,1100000,1350000]) {
  const elements = new Map();
  const context = new Proxy({}, {get:(_,key)=>key==='measureText'?()=>({width:100}):()=>{},set:()=>true});
  function element() {return {value:'1',checked:true,children:[],textContent:'',width:800,height:960,
    classList:{toggle(){}},addEventListener(name,fn){this[name]=fn;},getContext(){return context;},
    replaceChildren(){this.children=[];},append(el){this.children.push(el);},remove(){},};}
  const get=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id);};
  const frames = timestamps.map((timestamp,index)=>({index,timestamp,width:1920,height:1280,
    points:[],totalPoints:100,boxes:[],timings:[1,2,3,4,5,0,7,8]}));
  let now=0, animation, keydown;
  const sandbox={window:{sceneManifest:{timestamps,segment:'test',source:'YOLOX ONNX predictions'}},
    document:{getElementById:get,createElement:element,head:{append(script){queueMicrotask(()=>{
      const index=Number(script.src.match(/frame-(\d+)/)[1]);sandbox.window.receiveFrame({...frames[index]});
    });}},addEventListener(name,fn){keydown=fn;}},
    Image:class{async decode(){}},performance:{now:()=>now},requestAnimationFrame:fn=>{animation=fn;}};
  vm.runInNewContext(source,sandbox);
  return {get, async flush(){await new Promise(r=>setImmediate(r));},tick(t){now=t;animation(t);},
    key(code){keydown({code,target:{tagName:'BODY'},preventDefault(){}});}};
}
test('scrub chooses sensor timestamps, not a nominal frame rate; stepping clamps',async()=>{
  const p=player();await p.flush();assert.equal(p.get('frame').textContent,'Frame 1 / 3');
  p.get('timeline').value='0.3';p.get('timeline').oninput();await p.flush();
  assert.equal(p.get('frame').textContent,'Frame 2 / 3');
  p.key('ArrowRight');await p.flush();assert.equal(p.get('frame').textContent,'Frame 3 / 3');
  p.key('ArrowRight');await p.flush();assert.equal(p.get('frame').textContent,'Frame 3 / 3');
  assert.match(p.get('time').textContent,/0.35 \/ 0.35 s/);
});
test('playback speed, non-looping end, restart and pause',async()=>{
  const p=player();await p.flush();p.get('loop').checked=false;p.get('speed').value='2';
  p.get('play').onclick();p.tick(60);await p.flush();assert.equal(p.get('frame').textContent,'Frame 2 / 3');
  p.tick(200);await p.flush();assert.equal(p.get('frame').textContent,'Frame 3 / 3');assert.equal(p.get('play').textContent,'Play');
  p.key('Space');p.tick(201);await p.flush();assert.equal(p.get('frame').textContent,'Frame 1 / 3');
  p.key('Space');assert.equal(p.get('play').textContent,'Play');
});
test('a single-frame scene stops cleanly and rapid seeks retain latest selection',async()=>{
  const one=player([1000000]);await one.flush();one.get('play').onclick();one.tick(10);await one.flush();assert.equal(one.get('play').textContent,'Play');
  const p=player();p.get('next').onclick();p.get('next').onclick();await p.flush();assert.equal(p.get('frame').textContent,'Frame 3 / 3');
});
