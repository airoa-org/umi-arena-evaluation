"""Episode checkpoints and self-contained diagnostic reports."""

import base64
from io import BytesIO
import json

from PIL import Image


def thumbnail(array):
    image = Image.fromarray(array)
    image.thumbnail((320, 240))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=75)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def _write_json(path, value):
    data = json.dumps(value, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data + "\n")
    temporary.replace(path)


def write_episode(output, episode):
    directory = output / "episodes"
    directory.mkdir(exist_ok=True)
    path = directory / f"episode-{episode['episode_index']}-repeat-{episode['repeat']}.json"
    _write_json(path, episode)


def write_metadata(output, report):
    _write_json(output / "metadata.json", {key: value for key, value in report.items() if key != "episodes"})


def write_report(output, report):
    data = json.dumps(report, allow_nan=False)
    temporary = output / "report.json.tmp"
    temporary.write_text(data + "\n")
    temporary.replace(output / "report.json")
    temporary = output / "report.html.tmp"
    temporary.write_text(HTML.replace("__REPORT__", data.replace("<", "\\u003c")))
    temporary.replace(output / "report.html")


HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>UMI replay evaluation</title>
<style>
:root{color-scheme:light;--ink:#182638;--muted:#53647a;--line:#dce3ed;--blue:#2563a6;--orange:#b9501d}
*{box-sizing:border-box}body{font:15px system-ui,sans-serif;background:#f4f6fa;color:var(--ink);max-width:1320px;margin:32px auto;padding:0 24px}
h1{margin:0 0 6px}h2{font-size:19px;margin:0 0 12px}h3{font-size:15px;margin:16px 0 8px}p{line-height:1.5;margin:10px 0}small,.muted{color:var(--muted)}a{color:var(--blue)}
.card{background:white;border:1px solid var(--line);border-radius:12px;padding:22px;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.grid .card{margin:0;min-width:0}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:16px 0}.metric{background:#f5f8fc;border:1px solid var(--line);border-radius:9px;padding:14px}.metric strong{display:block;font-size:24px;margin:7px 0}.metric:first-child{background:#edf4ff}.metric .pair{display:flex;gap:16px;flex-wrap:wrap}.pair b{display:block;font-size:21px;margin:6px 0}.left{color:var(--blue)}.right{color:var(--orange)}
.controls{display:flex;align-items:end;gap:12px;flex-wrap:wrap}.controls label{display:flex;flex-direction:column;gap:6px;min-width:150px;flex:1}.controls .subtask{flex:2;min-width:240px}
select,button{font:inherit;color:inherit;border:1px solid #a9b6c8;background:white;border-radius:6px;padding:9px;max-width:100%}button,summary{cursor:pointer}button:hover{background:#edf4ff}button:disabled{opacity:.45;cursor:default}
:focus-visible{outline:3px solid #3977cd;outline-offset:3px}input[type=range]{width:100%;accent-color:var(--blue)}input[type=checkbox]{accent-color:var(--blue)}.toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:12px 0}.toolbar output{margin-right:auto;font-weight:650}.text-button{padding:0;border:0;background:none;text-align:left;color:var(--blue)}
details{margin-top:14px}summary{font-weight:600}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid var(--line)}th{font-size:13px;color:var(--muted)}.table-wrap{overflow-x:auto}.table-wrap table{min-width:660px}.selected-row{background:#f1f6ff}
svg{width:100%;height:auto;display:block}.legend{display:flex;gap:8px 16px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}.legend span{display:inline-flex;align-items:center;gap:5px}.legend svg{width:32px;height:14px}.readout{display:block;min-height:36px;font-size:12px;color:var(--muted);margin-top:10px}
.contribution{display:flex;height:10px;border-radius:5px;overflow:hidden;background:#edf0f5;margin:10px 0}.contribution span{min-width:0}.contribution-labels{display:flex;gap:16px;flex-wrap:wrap;font-size:13px}.contribution-labels i{display:inline-block;width:9px;height:9px;margin-right:5px;border-radius:2px}
#timeline svg{max-height:125px}#timeline-readout{font-size:13px;min-height:20px}.spread{width:240px;max-width:100%}.spread svg{overflow:visible}.spread text{font-size:11px}.spread [role=button]{cursor:pointer}.spread [role=button]:hover{fill:#edf4ff}.recordings{display:flex;gap:24px;align-items:center;flex-wrap:wrap}.recordings p{max-width:450px}
#images{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;max-width:calc(50% - 6px)}figure img{display:block;width:280px;max-width:100%;border-radius:8px}figcaption{font-size:12px;color:var(--muted);margin:5px 0}.error{color:#a42b2b}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}[hidden]{display:none!important}
@media(max-width:760px){body{padding:0 12px;margin:20px auto}.card{padding:16px}.metrics{grid-template-columns:1fr 1fr}.grid{grid-template-columns:1fr}.controls label,.controls .subtask{min-width:100%;flex:1}.metric strong{font-size:22px}.pair b{font-size:19px}}
</style>
<header><h1>UMI replay evaluation</h1><p id="run-label"></p></header>
<section class="card" aria-labelledby="verdict"><h2 id="verdict"></h2><pre id="warnings" class="error" hidden></pre><p id="coverage" class="muted"></p><p id="reference-filter" class="muted" hidden></p>
<div id="overall-metrics" class="metrics"></div><p id="secondary" class="muted"></p>
<h3>What drives the overall error?</h3><div id="overall-contributions"></div>
<details><summary>How is this calculated?</summary><p id="calculation"></p><p id="weighting"></p></details>
<p class="muted">Recorded observations measure demonstration agreement. Predicted actions do not change these images; this is not a measure of robot task completion.</p></section>
<section class="card" aria-labelledby="explore"><h2 id="explore">Explore a subtask</h2>
<div class="controls"><label>Task<select id="task"></select></label><label class="subtask">Subtask<select id="subtask"></select></label>
<label>Recording<select id="recording"></select></label><label id="repeat-label">Repeat<select id="repeat"></select></label></div>
<h3 id="subtask-title"></h3><div id="subtask-metrics" class="metrics"></div>
<div class="recordings"><div id="recording-spread" class="spread"></div><p class="muted">Each dot is one recording’s normalized RMSE. Click a row to inspect it. Repeated runs are combined by averaging squared errors.</p></div>
<details id="task-details"><summary id="task-heading">Compare subtasks</summary><div id="hierarchy" class="table-wrap"></div></details></section>
<section class="card" id="inspector" aria-labelledby="recording-title"><h2 id="recording-title"></h2><p id="instruction"></p>
<div id="timeline"></div><p id="timeline-readout" class="muted">Select an error peak to inspect its chunk.</p>
<label for="window">Chunk in this recording</label><input id="window" type="range" min="0" value="0" step="1">
<div class="toolbar"><button id="previous">← Previous</button><output id="chunk-count" aria-live="polite"></output><button id="next">Next →</button><button id="worst">Highest-error chunk</button></div>
<p id="error" class="error" role="status"></p><p id="chunk-summary"></p><p id="chunk-errors" class="muted"></p><div id="chunk-contributions"></div><p id="time" class="muted"></p>
<div id="images"></div></section>
<div class="toolbar"><label><input id="lock-axes" type="checkbox"> Lock plot axes across this recording</label><small>Trajectory plots use equal centimetre scales. Hover for values; focus a plot and use ← / → to inspect points.</small></div>
<div class="grid" id="plots"></div>
<section class="card"><details><summary>Dataset IDs, evaluation settings and reference audit</summary><pre id="settings"></pre></details></section>
<script id="data" type="application/json">__REPORT__</script>
<script>
const report=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const fmt=(v,d=2)=>v==null?'—':Number(v).toFixed(d), mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const keys=['position_cm','rotation_deg','gripper_rad'], names=['Position','Orientation','Gripper'];
const scales=[report.settings.position_scale_cm,report.settings.rotation_scale_deg,report.settings.gripper_scale_rad];
const colors=['#2563a6','#7954a2','#b9501d'], hands=['Left','Right'], handColors=['#2563a6','#b9501d'];
const suiteTasks=report.suite?.tasks||[];
const entries=(report.episodes||[]).map((e,index)=>{
 const task=suiteTasks.find(t=>t.recordings.some(r=>r.episodes.includes(e.episode_index)));
 const recording=task?.recordings.find(r=>r.episodes.includes(e.episode_index)), step=e.step??(recording?recording.episodes.indexOf(e.episode_index)+1:0);
 return {e,index,task:String(task?.id??e.task??'Unassigned'),taskName:task?.name||e.task||'Unassigned',step:String(step),
  instruction:e.subtask||task?.steps[step-1]||'Selected episodes',recording:recording?.uuid||String(e.episode_index)};
});
let selected=null, chunk=0;
function node(tag,text,parent,cls){const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;if(parent)parent.append(n);return n}
function svgNode(tag,attrs,parent,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!=null)n.textContent=text;if(parent)parent.append(n);return n}
function unique(list,key){return [...new Map(list.map(v=>[v[key],v])).values()]}
function options(id,items,value){const el=$(id);el.replaceChildren();items.forEach(([v,t])=>{const o=node('option',t,el);o.value=v});el.value=items.some(([v])=>v===value)?value:(items[0]?.[0]??'');el.disabled=!items.length;return el.value}
function complete(list){return list.filter(m=>m.e.status==='complete'&&m.e.summary)}
function recordingGroups(list){return unique(list,'recording').map(m=>{const runs=list.filter(v=>v.recording===m.recording), valid=complete(runs);
 return {runs,value:valid.length===runs.length&&valid.length===report.repeats?Math.sqrt(mean(valid.map(v=>v.e.summary.normalized_mse))):null}})}
function combined(list){const groups=recordingGroups(list),task=suiteTasks.find(t=>String(t.id)===list[0]?.task);
 if(!groups.length||groups.some(g=>g.value==null)||(task&&groups.length!==task.recordings.length)||(!task&&report.status!=='complete'))return null;
 const perGroup=groups.map(g=>({normalized_mse:mean(g.runs.map(v=>v.e.summary.normalized_mse)),
  score:mean(g.runs.map(v=>v.e.summary.score)),...Object.fromEntries(keys.map(k=>[k,{mse:[0,1].map(h=>mean(g.runs.map(v=>v.e.summary[k].mse[h])))}]))}));
 const s={normalized_rmse:Math.sqrt(mean(perGroup.map(v=>v.normalized_mse))),score:mean(perGroup.map(v=>v.score))};
 keys.forEach(k=>{const mse=[0,1].map(h=>mean(perGroup.map(v=>v[k].mse[h])));s[k]={mse,rmse:mse.map(Math.sqrt)}});return s}
function metricCards(id,s){const target=$(id);target.replaceChildren();if(!s){node('p','No complete aggregate available.',target);return}
 const first=node('div',null,target,'metric');node('small','Combined error · normalized RMSE',first);node('strong',fmt(s.normalized_rmse),first);node('small','Lower is better · 0 is perfect',first);
 keys.forEach((k,i)=>{const card=node('div',null,target,'metric');node('small',names[i]+' RMSE · '+['cm','degrees','rad'][i],card);const pair=node('div',null,card,'pair');
  hands.forEach((hand,h)=>{const value=node('span',null,pair,h?'right':'left');node('small',hand,value);node('b',fmt(s[k].rmse[h],i===2?3:2),value)})})}
function contributions(id,s){const target=$(id);target.replaceChildren();if(!s)return;
 const parts=keys.map((k,i)=>s[k].mse.reduce((a,b)=>a+b,0)/(6*scales[i]*scales[i])),total=parts.reduce((a,b)=>a+b,0);
 if(total<1e-20){node('small','Error is effectively zero; no meaningful contribution breakdown.',target);return}
 const bar=node('div',null,target,'contribution'),labels=node('div',null,target,'contribution-labels');
 parts.forEach((v,i)=>{const percent=100*v/total,segment=node('span',null,bar);segment.style.width=percent+'%';segment.style.background=colors[i];
  const label=node('span',null,labels);node('i',null,label).style.background=colors[i];node('span',names[i]+' '+fmt(percent,1)+'%',label)});
 target.title='Shares of normalized squared error, before taking the square root. These are not percentages of task success.'}
function selectEntry(index,worst=false){const m=entries[index];if(!m)return;sync({task:m.task,step:m.step,recording:m.recording,repeat:String(m.e.repeat)});if(worst)showWorst();$('inspector').scrollIntoView({block:'start'})}
function spread(target,list,max){target.replaceChildren();const groups=recordingGroups(list),scale=max||Math.max(1,...groups.map(g=>g.value||0));
 const svg=svgNode('svg',{viewBox:'0 0 240 '+(groups.length*23+24),role:'group','aria-label':'Recording errors; lower is better'},target);
 groups.forEach((g,i)=>{const label='Recording '+(i+1)+': '+(g.value==null?'incomplete':'normalized RMSE '+fmt(g.value));
  const active=g.runs[0].recording===$('recording').value,row=svgNode('g',{role:'button',tabindex:0,'aria-label':label,'aria-pressed':active},svg);svgNode('title',{},row,label);
  svgNode('rect',{x:0,y:i*23,width:240,height:23,fill:active?'#edf4ff':'transparent'},row);svgNode('text',{x:2,y:16+i*23,fill:'#53647a'},row,'R'+(i+1));
  svgNode('line',{x1:30,x2:183,y1:12+i*23,y2:12+i*23,stroke:'#dce3ed'},row);
  if(g.value!=null)svgNode('circle',{cx:30+153*g.value/scale,cy:12+i*23,r:4,fill:'#2563a6'},row);
  svgNode('text',{x:192,y:16+i*23,fill:'#182638'},row,fmt(g.value));
  const activate=()=>selectEntry(g.runs.find(m=>m.e.repeat===Number($('repeat').value))?.index??g.runs[0].index);
  row.onclick=activate;row.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();activate()}}});
 svgNode('text',{x:30,y:groups.length*23+17,fill:'#53647a'},svg,'0');svgNode('text',{x:168,y:groups.length*23+17,fill:'#53647a'},svg,fmt(scale))}
function taskTable(){const target=$('hierarchy');target.replaceChildren();const members=entries.filter(m=>m.task===$('task').value), task=report.aggregate?.tasks?.find(t=>String(t.id)===$('task').value);
 $('task-heading').textContent='Compare all subtasks · '+(members[0]?.taskName||'No episodes')+(task?' · error '+fmt(task.summary.normalized_rmse):'');
 const steps=unique(members,'step'),table=node('table',null,target),head=node('tr',null,table);
 ['Subtask','Normalized RMSE ↓','Recording variation',''].forEach(t=>node('th',t,head));
 const max=Math.max(1,...recordingGroups(members).map(g=>g.value||0),...members.map(m=>m.e.summary?.normalized_rmse||0));
 steps.forEach(m=>{const list=members.filter(v=>v.step===m.step),s=combined(list),row=node('tr',null,table,m.step===$('subtask').value?'selected-row':'');
  const cell=node('td',null,row),button=node('button',(m.step!=='0'?m.step+'. ':'')+m.instruction,cell,'text-button');button.onclick=()=>selectEntry(list[0].index);
  node('td',s?fmt(s.normalized_rmse):'Incomplete',row);const dots=node('div',null,node('td',null,row),'spread');spread(dots,list,max);
  const inspect=node('button','Inspect worst',node('td',null,row));inspect.title='Open the highest-error recording/repeat and its worst chunk';
  const runs=complete(list);inspect.disabled=!runs.length;inspect.onclick=()=>selectEntry(runs.reduce((a,b)=>a.e.summary.normalized_mse>=b.e.summary.normalized_mse?a:b).index,true)})}
function sync(preferred={}){const task=options('task',unique(entries,'task').map(m=>[m.task,m.taskName]),preferred.task??$('task').value);
 const members=entries.filter(m=>m.task===task),step=options('subtask',unique(members,'step').map(m=>[m.step,(m.step!=='0'?m.step+'. ':'')+m.instruction]),preferred.step??$('subtask').value);
 const list=members.filter(m=>m.step===step),groups=recordingGroups(list),recording=options('recording',groups.map((g,i)=>[g.runs[0].recording,'Recording '+(i+1)+' of '+groups.length]),preferred.recording??$('recording').value);
 const runs=list.filter(m=>m.recording===recording),repeat=options('repeat',runs.map(m=>[String(m.e.repeat),'Repeat '+(m.e.repeat+1)+' · '+m.e.status]),preferred.repeat??$('repeat').value);
 $('repeat-label').hidden=report.repeats===1;selected=runs.find(m=>String(m.e.repeat)===repeat)||null;chunk=0;
 $('subtask-title').textContent='Selected subtask · '+(list[0]?.instruction||'No episodes');metricCards('subtask-metrics',combined(list));spread($('recording-spread'),list);taskTable();render()}
function bounds(series,spatial=false,zero=false){const points=series.flatMap(s=>s.points);let x0=Infinity,x1=-Infinity,y0=Infinity,y1=-Infinity;
 points.forEach(([x,y])=>{x0=Math.min(x0,x);x1=Math.max(x1,x);y0=Math.min(y0,y);y1=Math.max(y1,y)});
 if(zero){x0=0;y0=0}if(x1-x0<1e-8){x0-=.5;x1+=.5}if(y1-y0<1e-8){if(!zero)y0-=.5;y1+=.5}
 const px=(x1-x0)*.04,py=(y1-y0)*.08;if(!zero)x0-=px;x1+=px;if(!zero)y0-=py;y1+=py;
 if(spatial){const unit=Math.max((x1-x0)/500,(y1-y0)/220),xm=(x0+x1)/2,ym=(y0+y1)/2;x0=xm-unit*250;x1=xm+unit*250;y0=ym-unit*110;y1=ym+unit*110}
 return {x0,x1,y0,y1}}
function marker(svg,x,y,h,filled,color,r=4){return h?svgNode('path',{d:'M'+x+','+(y-r)+'L'+(x+r)+','+y+'L'+x+','+(y+r)+'L'+(x-r)+','+y+'Z',fill:filled?color:'white',stroke:color,'stroke-width':1.5},svg):svgNode('circle',{cx:x,cy:y,r,fill:filled?color:'white',stroke:color,'stroke-width':1.5},svg)}
function plot(title,series,xlabel,ylabel,pool,spatial=false,zero=false){const b=bounds(pool,spatial,zero),xp=x=>65+500*(x-b.x0)/(b.x1-b.x0),yp=y=>250-220*(y-b.y0)/(b.y1-b.y0);
 const card=node('div',null,$('plots'),'card');node('h2',title,card);const svg=svgNode('svg',{viewBox:'0 0 600 310',role:'img',tabindex:0,'aria-label':title+'; use arrow keys to inspect values','data-bounds':JSON.stringify(b)},card);
 svgNode('path',{d:'M65 30V250H565',fill:'none',stroke:'#9aabba'},svg);
 for(let t=0;t<=4;t++){const x=b.x0+(b.x1-b.x0)*t/4,y=b.y0+(b.y1-b.y0)*t/4;
  svgNode('line',{x1:65,x2:565,y1:yp(y),y2:yp(y),stroke:'#edf0f5'},svg);svgNode('text',{x:xp(x),y:273,'text-anchor':'middle',fill:'#53647a','font-size':11},svg,fmt(x));
  svgNode('text',{x:57,y:yp(y)+4,'text-anchor':'end',fill:'#53647a','font-size':11},svg,fmt(y))}
 svgNode('text',{x:315,y:300,'text-anchor':'middle',fill:'#53647a','font-size':12},svg,xlabel);svgNode('text',{x:65,y:18,fill:'#53647a','font-size':12},svg,ylabel);
 series.forEach(s=>{svgNode('path',{d:s.points.map((p,i)=>(i?'L':'M')+xp(p[0])+','+yp(p[1])).join(' '),fill:'none',stroke:s.color,'stroke-width':2,'stroke-dasharray':s.dash||''},svg);
  marker(svg,xp(s.points[0][0]),yp(s.points[0][1]),s.hand,false,s.color);const end=s.points.at(-1);marker(svg,xp(end[0]),yp(end[1]),s.hand,true,s.color)});
 const legend=node('div',null,card,'legend');series.forEach(s=>{const item=node('span',null,legend),key=svgNode('svg',{viewBox:'0 0 32 14','aria-hidden':true},item);
  svgNode('line',{x1:0,x2:32,y1:7,y2:7,stroke:s.color,'stroke-width':2,'stroke-dasharray':s.dash||''},key);marker(key,16,7,s.hand,true,s.color,3);node('span',s.name,item)});
 node('small','Hollow marker = first point · filled marker = last point',card);
 const readout=node('output','Hover a point, or use ← / → while this plot is focused.',card,'readout');readout.setAttribute('aria-live','polite');let point=0;
 const cursor=svgNode('g',{'aria-hidden':true},svg);
 function inspect(index){point=Math.max(0,Math.min(index,series[0].points.length-1));cursor.replaceChildren();
  const labels=series.map(s=>{const p=s.points[Math.min(point,s.points.length-1)];marker(cursor,xp(p[0]),yp(p[1]),s.hand,false,s.color,6);return s.name+': '+fmt(p[0])+' '+xlabel+', '+fmt(p[1],3)+' '+ylabel});
  readout.textContent='Point '+(point+1)+' · '+labels.join(' · ')}
 svg.onpointermove=event=>{const p=svg.createSVGPoint();p.x=event.clientX;p.y=event.clientY;const local=p.matrixTransform(svg.getScreenCTM().inverse());let closest=Infinity,index=0;
  series.forEach(s=>s.points.forEach((v,i)=>{const d=(xp(v[0])-local.x)**2+(yp(v[1])-local.y)**2;if(d<closest){closest=d;index=i}}));inspect(index)};
 svg.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();inspect(point+(e.key==='ArrowRight'?1:-1))}}}
function trajectory(w,axis){return [0,1].flatMap(h=>[['reference_poses','recorded','5 4'],['predicted_poses','predicted','']].map(([key,label,dash])=>({name:hands[h]+' '+label,hand:h,color:handColors[h],dash,points:w[key].map(v=>[100*v[h][0],100*v[h][axis]])})))}
function errors(w,key){return [0,1].map(h=>({name:hands[h],hand:h,color:handColors[h],points:w[key].map((v,i)=>[(i+1)/report.settings.action_hz,v[h]])}))}
function grippers(w){return [0,1].flatMap(h=>[['reference_grippers','recorded','5 4'],['predicted_grippers','predicted','']].map(([key,label,dash])=>({name:hands[h]+' '+label,hand:h,color:handColors[h],dash,points:w[key].map((v,i)=>[(i+1)/report.settings.action_hz,v[h]])})))}
function timeline(e){$('timeline').replaceChildren();if(!e?.windows?.length)return;const values=e.windows.map(w=>w.summary.normalized_rmse),max=Math.max(1,...values),start=e.windows[0].timestamp,end=e.windows.at(-1).timestamp;
 const width=Math.max(320,Math.min(1000,$('timeline').clientWidth)),x=i=>35+(width-70)*(end>start?(e.windows[i].timestamp-start)/(end-start):.5),y=v=>85-65*v/max;
 const svg=svgNode('svg',{viewBox:'0 0 '+width+' 115',role:'group','aria-label':'Chunk error timeline'},$('timeline'));
 svgNode('path',{d:values.map((v,i)=>(i?'L':'M')+x(i)+','+y(v)).join(' '),fill:'none',stroke:'#2563a6','stroke-width':2},svg);
 svgNode('line',{x1:35,x2:width-35,y1:85,y2:85,stroke:'#dce3ed'},svg);
 [0,max].forEach(v=>svgNode('text',{x:28,y:y(v)+4,'text-anchor':'end',fill:'#53647a','font-size':11},svg,fmt(v,1)));
 svgNode('text',{x:35,y:12,fill:'#53647a','font-size':11},svg,'Chunk normalized RMSE · lower is better');
 [[35,fmt(start)+' s'],[width-80,fmt(end)+' s']].forEach(([px,label])=>svgNode('text',{x:px,y:109,fill:'#53647a','font-size':11},svg,label));
 values.forEach((v,i)=>{const label='Chunk '+(i+1)+' · '+fmt(e.windows[i].timestamp)+' s · error '+fmt(v),dot=svgNode('circle',{cx:x(i),cy:y(v),r:i===chunk?6:4,fill:i===chunk?'#182638':'#2563a6',stroke:'white','stroke-width':1.5,role:'button',tabindex:-1,'aria-label':label,'aria-pressed':i===chunk},svg);
  svgNode('title',{},dot,label);dot.onclick=()=>{chunk=i;render()};dot.onpointerenter=()=>{$('timeline-readout').textContent=label}})}
function render(){const e=selected?.e;$('plots').replaceChildren();$('images').replaceChildren();$('chunk-contributions').replaceChildren();
 $('error').textContent=report.error||e?.error||'';$('chunk-summary').textContent='';$('chunk-errors').textContent='';
 $('settings').textContent=JSON.stringify({episode_index:e?.episode_index,uuid:e?.uuid,repeat:e?.repeat,settings:report.settings,dataset_revision:report.dataset_revision,checkpoint:report.checkpoint_id,audit:e?.audit,gripper_check:report.gripper_check,server_reset:report.server_reset,suite_sha256:report.suite_sha256},null,2);
 const windows=e?.windows||[];chunk=Math.max(0,Math.min(chunk,windows.length-1));$('window').max=Math.max(0,windows.length-1);$('window').value=chunk;
 $('window').disabled=!windows.length;$('previous').disabled=!windows.length||chunk===0;$('next').disabled=!windows.length||chunk===windows.length-1;$('worst').disabled=!windows.length;
 $('recording-title').textContent=selected?$('recording').selectedOptions[0].textContent+' · '+(e.status==='complete'?'normalized RMSE '+fmt(e.summary.normalized_rmse):e.status):'No recording available';
 $('instruction').textContent=selected?.instruction||'';$('chunk-count').textContent=windows.length?'Chunk '+(chunk+1)+' of '+windows.length:'No chunks available';timeline(e);
 $('timeline-readout').textContent=windows.length?'Select an error peak, or use the slider and previous/next buttons.':'No trajectory available';
 if(!windows.length){$('time').textContent='';return}const w=windows[chunk],s=w.summary,pool=$('lock-axes').checked?windows:[w];
 $('instruction').textContent=w.prompt;$('chunk-summary').textContent='This chunk · normalized RMSE '+fmt(s.normalized_rmse)+' · '+fmt(s.score)+'% within tolerance · '+s.timesteps+' valid actions';
 $('chunk-errors').textContent='Chunk RMSE (left / right): '+keys.map((k,i)=>names[i]+' '+s[k].rmse.map(v=>fmt(v,i===2?3:2)).join(' / ')+' '+['cm','°','rad'][i]).join(' · ');
 contributions('chunk-contributions',s);$('time').textContent='Observation '+fmt(w.timestamp,3)+' s · '+fmt(w.latency_ms,1)+' ms inference · predictions span '+fmt(s.timesteps/report.settings.action_hz,3)+' s';
 (w.images||[]).forEach((src,i)=>{const figure=node('figure',null,$('images')),img=node('img',null,figure);img.src=src;img.alt=hands[i]+' wrist recorded observation';node('figcaption',hands[i]+' wrist · recorded observation',figure)});
 if(!w.images?.length)node('p','No observation images saved for this run.',$('images'),'muted');
 for(const [axis,label] of [[1,'Y'],[2,'Z']])plot('Hand trajectories · X / '+label,trajectory(w,axis),'X (cm)',label+' (cm)',pool.flatMap(v=>trajectory(v,axis)),true);
 keys.forEach((key,i)=>plot(names[i]+' error',errors(w,key),'Time from chunk anchor (s)',['cm','degrees','radians'][i],pool.flatMap(v=>errors(v,key)),false,true));
 plot('Gripper commands',grippers(w),'Future time from observation (s)','radians',pool.flatMap(grippers))}
function showWorst(){const windows=selected?.e.windows||[];if(!windows.length)return;chunk=windows.reduce((best,w,i)=>w.summary.normalized_mse>windows[best].summary.normalized_mse?i:best,0);render()}
$('task').onchange=()=>sync({step:'',recording:'',repeat:''});$('subtask').onchange=()=>sync({recording:'',repeat:''});$('recording').onchange=()=>sync();$('repeat').onchange=()=>sync();
$('window').oninput=()=>{chunk=Number($('window').value);render()};$('previous').onclick=()=>{chunk--;render()};$('next').onclick=()=>{chunk++;render()};$('worst').onclick=showWorst;$('lock-axes').onchange=render;
let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>timeline(selected?.e),100)});
$('run-label').textContent=(report.checkpoint_id||report.baseline||'Replay run')+' · '+report.settings.action_hz+' Hz · demonstration agreement';
$('verdict').textContent=report.status==='complete'?'Evaluation finished · overall results':'Evaluation '+report.status+' · overall results unavailable';
$('warnings').hidden=!report.warnings?.length;$('warnings').textContent=(report.warnings||[]).join('\\n');
$('coverage').textContent=report.coverage?report.coverage.selected_tasks.length+' of '+report.coverage.total_open_tasks+' open tasks · '+report.coverage.selected_steps+' subtask positions · '+report.requested_episodes.length+' episodes. Uncovered: '+report.coverage.uncovered_tasks.map(t=>t.name).join(', '):report.requested_episodes.length+' selected episodes · '+report.repeats+' repeat(s)';
if(report.suite?.gripper_filter){$('reference-filter').hidden=false;$('reference-filter').textContent='Reference selection: both grippers must stay at or above '+report.suite.gripper_filter.minimum_rad+' rad in observations and reference actions, across every episode of each recording. '+(report.gripper_check?.status==='passed'?'Raw gripper values checked in every selected episode.':'Gripper validation has not passed.')}
metricCards('overall-metrics',report.aggregate);contributions('overall-contributions',report.aggregate);
$('secondary').textContent=report.aggregate?'Within tolerance: '+fmt(report.aggregate.score)+'% of valid timesteps'+(report.aggregate.moving_score==null?'':', and '+fmt(report.aggregate.moving_score)+'% of the timesteps where a hand is moving')+', aggregated with the selection weights below. Secondary diagnostic.':'';
$('calculation').textContent='RMSE means root mean square error: square each error, average those squares, then take the square root. Position uses 3D distance, orientation uses the shortest rotation angle, and gripper uses joint-angle error. The combined error divides these by '+scales[0]+' cm, '+scales[1]+'°, and '+scales[2]+' rad before squaring, with equal weight for both hands and all three measurements. Zero is perfect; 1 is a reference level, not a pass cutoff. Reference scales are provisional. Within tolerance checks all six errors against '+report.settings.position_cm+' cm, '+report.settings.rotation_deg+'°, and '+report.settings.gripper_rad+' rad.';
$('weighting').textContent='Squared-error weighting: '+(report.aggregate?.weighting||'only complete selections receive an aggregate')+'. Take the square root after aggregation. Timing: '+report.settings.translation_frame+' translation, '+report.settings.pose_timing+' pose interval.';
sync();
</script></html>
"""
