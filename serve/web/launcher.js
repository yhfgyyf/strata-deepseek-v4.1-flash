"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="strata-token"]').content;
let current = null, dirty = false, logKind = "service", busy = false, lastJob = "";
const phaseNames = {queued:"Queued", dependencies:"Installing dependencies", virtualenv:"Creating environment", configuring:"Configuring build", building:"Building engine", downloading:"Downloading files", verifying:"Verifying files", verified:"Verification complete", validating:"Checking configuration", complete:"Setup complete", failed:"Setup failed", cancelled:"Setup cancelled"};
const stateNames = {stopped:"Stopped", starting:"Loading model", ready:"Model ready", stopping:"Stopping", exited:"Process exited", orphaned:"Process needs attention", failed:"Startup failed"};
function notice(message, success=false) { $("notice").textContent=message; $("notice").classList.toggle("success",success); $("notice").hidden=false; }
async function api(path, body) {
  const response=await fetch("/api/"+path,{method:body===undefined?"GET":"POST",headers:{"X-Strata-Token":token,"Content-Type":"application/json"},...(body===undefined?{}:{body:JSON.stringify(body)})});
  const value=await response.json();
  if(!response.ok || value.error) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
function edit() { dirty=true; $("dirty").textContent="Unsaved changes"; updateControls(); }
function fill(config) {
  const c=config.values,s=config.setup;
  for(const el of document.querySelectorAll("[data-config]")) el.value=c[el.dataset.config]||"";
  $("variant").value=c.variant; $("source").value=s.source; $("data-dir").value=s.data_dir;
  $("no-proxy").checked=s.no_proxy; $("vision").checked=!!c.vision; $("vision").disabled=c.variant.startsWith("GSQ");
  $("port").value=c.port||30011; $("host").value=c.host||"127.0.0.1";
  $("prefix-cache").value=c.prefix_cache_mib??0; $("hot-bytes").value=c.hot_bytes??"";
  $("cuda").value=s.cuda; $("build-dir").value=s.build_dir; $("ggml-dir").value=s.ggml_dir; $("jobs").value=s.jobs;
  $("api-key").value=""; $("clear-key").checked=false; $("key-status").textContent=config.has_api_key?"Set":"Not set";
  dirty=false; $("dirty").textContent=config.exists?"Configuration saved":"Save the configuration to get started";
}
function payload() {
  const set={variant:$("variant").value,port:Number($("port").value),host:$("host").value.trim(),prefix_cache_mib:Number($("prefix-cache").value),vision:$("vision").checked};
  for(const el of document.querySelectorAll("[data-config]")) set[el.dataset.config]=el.value.trim();
  const hot=$("hot-bytes").value.trim();
  if(hot) { if(hot!=="auto" && (!/^\d+$/.test(hot)||!Number.isSafeInteger(Number(hot)))) throw new Error("GPU budget must be auto or a valid integer number of bytes"); set.hot_bytes=hot==="auto"?hot:Number(hot); } else set.hot_bytes=null;
  if($("api-key").value) set.api_key=$("api-key").value;
  return {set,clear_api_key:$("clear-key").checked,setup:{source:$("source").value,data_dir:$("data-dir").value.trim(),no_proxy:$("no-proxy").checked,cuda:$("cuda").value.trim(),build_dir:$("build-dir").value.trim(),ggml_dir:$("ggml-dir").value.trim(),jobs:Number($("jobs").value)}};
}
function updateControls() {
  if(!current)return;
  const s=current.service,job=current.job,preparing=["queued","running"].includes(job.state),active=s.running||s.state==="orphaned";
  for(const id of ["save","use-default-paths","prepare","prepare-launch"]) $(id).disabled=busy||active||preparing;
  $("start").disabled=busy||active||preparing||dirty||!current.config.exists;
  $("doctor").disabled=busy||active||preparing;
  $("prepare").disabled ||=dirty||!current.config.exists; $("prepare-launch").disabled ||=dirty||!current.config.exists;
  $("stop").disabled=busy||!s.running; $("chat").classList.toggle("disabled",!s.ready);
  $("cancel-prepare").disabled=busy||!preparing||active;
  $("chat").href=s.ready?s.url:"#";
  for(const el of document.querySelectorAll("#config-form input,#config-form select")) el.disabled=active||preparing;
  $("vision").disabled ||= $("variant").value.startsWith("GSQ");
}
async function refresh() {
  const previousState=current?.service.state;
  current=await api("state");
  if(!$("config-form").dataset.loaded){fill(current.config);$("config-form").dataset.loaded="1";}
  const s=current.service,j=current.job;
  if(s.state==="stopped" && ["starting","ready","stopping"].includes(previousState))notice("Model stopped.",true);
  if(s.state==="ready" && previousState==="starting")notice("Model ready. Open chat to begin.",true);
  const jobKey=String(j.started)+":"+j.state;
  if(j.state==="failed"&&jobKey!==lastJob)notice("Setup failed. "+(j.error||j.message||"Check the setup log below."));
  lastJob=jobKey;
  $("status").className="status "+s.state; $("status").querySelector("span").textContent=stateNames[s.state]||s.state;
  $("current-model").textContent=current.config.values.variant; $("endpoint").textContent=s.url;
  const progressText=phaseNames[j.phase]||phaseNames[j.state]||"No setup job yet";
  $("progress-label").textContent=progressText+(j.file?" · "+j.file:"")+(j.error?" · "+j.error:"");
  $("progress").value=j.total?Math.min(100,100*j.bytes/j.total):j.state==="complete"?100:0;
  updateControls();
}
async function logs() { const value=await api("logs?kind="+logKind); const el=$("logs"),bottom=el.scrollTop+el.clientHeight>=el.scrollHeight-20; el.textContent=value.text||"No logs yet.";if(bottom)el.scrollTop=el.scrollHeight; }
async function action(fn) { if(busy)return;busy=true;updateControls();try{await fn();await refresh();await logs();}catch(e){notice(e.message);}finally{busy=false;updateControls();} }
$("config-form").addEventListener("input",edit);
$("variant").addEventListener("change",()=>{ if($("variant").value.startsWith("GSQ"))$("vision").checked=false; updateControls();notice("Model changed. Select Use model paths, or enter the existing files for this model."); });
$("config-form").addEventListener("submit",e=>{e.preventDefault();action(async()=>{const v=await api("config",payload());const state=await api("state");fill(state.config);notice("Configuration saved."+(v.backup?" A backup of the previous version was saved.":""),true);});});
$("use-default-paths").addEventListener("click",()=>action(async()=>{const v=payload();v.reset_paths=true;for(const key of ["model","draft","tokenizer","mmproj","engine","profile","heat","integrity"])delete v.set[key];await api("config",v);const state=await api("state");fill(state.config);notice("Default paths saved for the selected model and directory.",true);}));
$("plan").addEventListener("click",()=>action(async()=>{if(dirty)throw new Error("Save the configuration before reviewing the setup plan.");const p=await api("plan");const box=$("plan-view");box.replaceChildren();const title=document.createElement("strong");title.textContent=`To download: ${(p.download_bytes/2**30).toFixed(2)} GiB · ${p.assets.length} files`;box.append(title);const list=document.createElement("ul");for(const a of p.assets){const li=document.createElement("li");li.textContent=`${a.exists?(a.size_matches?"Needs verification":"Size mismatch; original file preserved"):a.partial_bytes?"Resume":"Download"} · ${a.path} · ${(a.bytes/2**30).toFixed(2)} GiB`;if(a.exists&&!a.size_matches)li.className="bad";list.append(li);}for(const d of p.disks){const li=document.createElement("li");li.textContent=`${d.path}: ${(d.free_bytes/2**30).toFixed(1)} GiB free${d.enough?"":"; insufficient space"}`;if(!d.enough)li.className="bad";list.append(li);}box.append(list);}));
$("doctor").addEventListener("click",()=>action(async()=>{const d=await api("doctor",{});notice(d.checks.map(c=>(c.ok?"✓ ":"✕ ")+c.name+"："+c.detail).join("\n"),d.ok);}));
for(const [id,launch]of [["prepare",false],["prepare-launch",true]])$(id).addEventListener("click",()=>action(async()=>{await api("install",{launch});logKind="install";selectLog();notice("Setup started. Follow progress and logs below.",true);}));
$("start").addEventListener("click",()=>action(async()=>{await api("start",{});logKind="service";selectLog();notice("Model loading. Open chat when it is ready.",true);}));
$("stop").addEventListener("click",()=>action(async()=>{await api("stop",{});notice("Stop requested. The model status will update automatically.",true);}));
$("cancel-prepare").addEventListener("click",()=>action(async()=>{await api("cancel",{});notice("Setup cancelled. Partial downloads and logs were kept.",true);}));
function selectLog(){ $("service-log").classList.toggle("selected",logKind==="service");$("install-log").classList.toggle("selected",logKind==="install"); }
for(const kind of ["service","install"])$(kind+"-log").addEventListener("click",()=>{logKind=kind;selectLog();logs().catch(e=>notice(e.message));});
$("refresh-log").addEventListener("click",()=>logs().catch(e=>notice(e.message)));
refresh().then(logs).catch(e=>notice(e.message));
setInterval(()=>{if(!busy)refresh().then(logs).catch(e=>notice("Manager connection failed: "+e.message));},2000);
