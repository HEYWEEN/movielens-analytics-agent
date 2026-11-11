const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const { randomUUID } = require('node:crypto');

// A small DOM double exercises the app's actual send/restore logic without a
// network, browser, API key, or generated model reply.
class Element {
  constructor(tag='div') { this.tagName=tag;this.children=[];this.dataset={};this.style={};this.hidden=false;this.value='';this.options=[];this.className='';this.attributes={};this.parts=new Map();this.listeners=new Map();this.isConnected=true;this.classList={add:()=>{},remove:()=>{},contains:()=>false,toggle:()=>{}}; }
  set innerHTML(value) { this.html=value;this.children=[];this.parts.clear(); }
  get innerHTML() { return this.html || ''; }
  append(...nodes) { for(const node of nodes){node.parent=this;this.children.push(node);} }
  appendChild(node) {this.append(node);return node;}
  replaceChildren(...nodes) {this.children=[];this.append(...nodes);}
  querySelector(selector) { if(!this.parts.has(selector))this.parts.set(selector,new Element());return this.parts.get(selector); }
  querySelectorAll() { return []; }
  setAttribute(key,value) {this.attributes[key]=value;}
  removeAttribute(key) {delete this.attributes[key];}
  addEventListener(type,listener) {if(!this.listeners.has(type))this.listeners.set(type,new Set());this.listeners.get(type).add(listener);}
  removeEventListener(type,listener) {this.listeners.get(type)?.delete(listener);}
  scrollIntoView() {}
  focus() {}
  remove() {if(this.parent)this.parent.children=this.parent.children.filter(node=>node!==this);this.isConnected=false;}
}
function app() {
  const elements=new Map(),storage=new Map();
  const context=vm.createContext({console,crypto:{randomUUID},setInterval:()=>1,clearInterval(){},setTimeout,clearTimeout,
    requestAnimationFrame:()=>1,cancelAnimationFrame(){},TextDecoder,Uint8Array,
    document:{getElementById(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement:tag=>new Element(tag),querySelector:()=>new Element(),querySelectorAll:()=>[],addEventListener(){}},
    localStorage:{getItem:key=>storage.get(key)||null,setItem:(key,value)=>storage.set(key,value),removeItem:key=>storage.delete(key)}});
  const html=fs.readFileSync('index.html','utf8'),script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
  vm.runInContext(script.slice(0,script.lastIndexOf('  renderPendingScores(); renderSessionList();')),context);
  vm.runInContext('poll=async()=>{};pollAnalysis=async()=>{};',context);
  const run=code=>vm.runInContext(code,context);
  return {run,elements,storage};
}
const prediction={status:'completed',task:'classification',user_id:635,movie_id:1,movie_title:'Toy Story',results:[{algorithm:'logistic_regression',positive_probability:.924,predicted_high_rating:true,decision_threshold:.3,test_metrics:{f1:.745}}]};
const clusters={status:'completed',task:'clustering',user_id:635,profile:{train_ratings:31,high_rate:.935,top_genres:[]},results:[{algorithm:'kmeans',cluster_id:0,metrics:{silhouette:.148},segment:{size:2515,top_genres:['Drama'],summary:'Drama',representative_movies:[]}}]};
async function send(app,result,question) {
  app.run(`streamAgent=async()=>(${JSON.stringify({reply:question+' answer',result})});$('prompt').value=${JSON.stringify(question)};`);
  await app.run('send()');
}

test('a second question keeps the first structured result on its own assistant message',async()=>{
  const a=app();await send(a,prediction,'prediction');await send(a,clusters,'clusters');
  const messages=JSON.parse(a.storage.get('movielensAgentSessionsV2'))[0].messages.filter(m=>m.role==='assistant');
  assert.equal(messages[0].analysis?.results?.[0]?.task,'classification');
  assert.equal(messages[1].analysis?.results?.[0]?.task,'clustering');
  assert.notEqual(messages[0].id,messages[1].id);
  assert.equal(a.run('resultViews.size'),2);
});

test('refresh and session switching restore both per-message result snapshots',async()=>{
  const a=app();await send(a,prediction,'prediction');await send(a,clusters,'clusters');
  const saved=a.storage.get('movielensAgentSessionsV2');
  a.run(`sessions=JSON.parse(${JSON.stringify(saved)});restoreSession(sessions[0].id);`);
  assert.equal(a.run('resultViews.size'),2);
  assert.equal(a.run("[...resultViews.values()].filter(view=>view.root.innerHTML.includes('Toy Story')).length"),1);
  assert.equal(a.run("[...resultViews.values()].filter(view=>view.root.innerHTML.includes('2,515')).length"),1);
});

test('switching a result tab affects only its own message and persists the choice',async()=>{
  const a=app();const report={source:{task_id:'source',clean_data_version:'clean'},configuration:{n_clusters:3},split:{},classification:{algorithms:{}},clustering:{eligible_users:5301,algorithms:{kmeans:{segments:[{size:5301}]}}},reduction:{original_dimensions:37,algorithms:{pca:{actual_dimensions:2}}}};
  await send(a,{task:'analysis_report',report},'report one');
  await send(a,{task:'analysis_report',report},'report two');
  const before=a.run('[...resultViews.values()][1].root.innerHTML');
  a.run("[...resultViews.values()][0].selectTab('clustering')");
  assert.equal(a.run('[...resultViews.values()][1].root.innerHTML'),before);
  const messages=JSON.parse(a.storage.get('movielensAgentSessionsV2'))[0].messages.filter(m=>m.role==='assistant');
  assert.equal(messages[0].analysis.tab,'clustering');
  assert.equal(messages[1].analysis.tab,'overview');
  const ids=a.run("[...resultViews.values()].flatMap(view=>[...view.root.innerHTML.matchAll(/\\bid=\"([^\"]+)\"/g)].map(match=>match[1]))");
  assert.equal(new Set(ids).size,ids.length);
});

test('legacy sessions retain all text and migrate the one existing result only once',()=>{
  const a=app();
  a.run(`sessions=[{id:'legacy',messages:[{role:'user',content:'first question'},{role:'assistant',content:'first answer'},{role:'user',content:'second question'},{role:'assistant',content:'second answer'}],lastResult:[${JSON.stringify(clusters)}]}];normalizeSession(sessions[0]);normalizeSession(sessions[0]);`);
  assert.equal(a.run('sessions[0].messages.length'),4);
  assert.equal(a.run('sessions[0].messages[1].content'),'first answer');
  assert.equal(a.run('sessions[0].messages.filter(message=>message.analysis).length'),1);
  assert.equal(a.run('sessions[0].messages[3].analysis.results[0].task'),'clustering');
  assert.equal(a.run('new Set(sessions[0].messages.map(message=>message.id)).size'),4);
});

test('a failed background task updates its own pending card and persists the reason',async()=>{
  const a=app();
  a.run("analysisTaskId='123456abcdef';lastAnalysisStatus='queued';streamAgent=async()=>({reply:'任务已提交',analysis_task_id:'123456abcdef'});$('prompt').value='start analysis';");
  await a.run('send()');
  a.run("renderAnalysisStatus({analysis_task_id:'123456abcdef',status:'failed',error:'fixture training failure'})");
  const message=JSON.parse(a.storage.get('movielensAgentSessionsV2'))[0].messages.at(-1);
  assert.equal(message.analysis.pending,false);
  assert.match(message.analysis.error,/fixture training failure/);
  assert.match(a.run('[...resultViews.values()][0].root.innerHTML'),/fixture training failure/);
  assert.match(a.run('[...resultViews.values()][0].root.innerHTML'),/未完成/);
});

test('reusing the task overview disposes the previous tab handlers',()=>{
  const a=app();
  a.run("createAnalysisView($('analysisResults'),{taskId:null,results:[]});createAnalysisView($('analysisResults'),{taskId:null,results:[]});");
  assert.equal(a.run("$('analysisResults').listeners.get('click').size"),1);
  assert.equal(a.run("$('analysisResults').listeners.get('keydown').size"),1);
});

test('numbered reply paragraphs preserve their ordinal values across blank lines',()=>{
  const a=app();const html=a.run("formatMessage('1. 第一项\\n\\n2. 第二项\\n\\n3. 第三项')");
  assert.match(html,/<ol start="2">/);
  assert.match(html,/<ol start="3">/);
});
