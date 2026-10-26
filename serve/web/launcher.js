"use strict";
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="strata-token"]').content;
let current = null, dirty = false, logKind = "service", busy = false, lastJob = "";
const phaseNames = {queued:"等待准备", dependencies:"安装依赖", virtualenv:"创建环境", configuring:"配置构建", building:"编译引擎", downloading:"下载文件", verifying:"校验文件", verified:"校验完成", validating:"检查配置", complete:"准备完成", failed:"准备失败", cancelled:"准备已取消"};
const stateNames = {stopped:"未启动", starting:"模型加载中", ready:"模型已就绪", stopping:"正在停止", exited:"进程已退出", orphaned:"进程需要检查", failed:"启动失败"};
function notice(message, success=false) { $("notice").textContent=message; $("notice").classList.toggle("success",success); $("notice").hidden=false; }
async function api(path, body) {
  const response=await fetch("/api/"+path,{method:body===undefined?"GET":"POST",headers:{"X-Strata-Token":token,"Content-Type":"application/json"},...(body===undefined?{}:{body:JSON.stringify(body)})});
  const value=await response.json();
  if(!response.ok || value.error) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
function edit() { dirty=true; $("dirty").textContent="有未保存的修改"; updateControls(); }
function fill(config) {
  const c=config.values,s=config.setup;
  for(const el of document.querySelectorAll("[data-config]")) el.value=c[el.dataset.config]||"";
  $("variant").value=c.variant; $("source").value=s.source; $("data-dir").value=s.data_dir;
  $("no-proxy").checked=s.no_proxy; $("vision").checked=!!c.vision; $("vision").disabled=c.variant.startsWith("GSQ");
  $("port").value=c.port||30011; $("host").value=c.host||"127.0.0.1";
  $("prefix-cache").value=c.prefix_cache_mib??0; $("hot-bytes").value=c.hot_bytes??"";
  $("cuda").value=s.cuda; $("build-dir").value=s.build_dir; $("ggml-dir").value=s.ggml_dir; $("jobs").value=s.jobs;
  $("api-key").value=""; $("clear-key").checked=false; $("key-status").textContent=config.has_api_key?"已设置":"未设置";
  dirty=false; $("dirty").textContent=config.exists?"配置已保存":"首次使用，请保存配置";
}
function payload() {
  const set={variant:$("variant").value,port:Number($("port").value),host:$("host").value.trim(),prefix_cache_mib:Number($("prefix-cache").value),vision:$("vision").checked};
  for(const el of document.querySelectorAll("[data-config]")) set[el.dataset.config]=el.value.trim();
  const hot=$("hot-bytes").value.trim();
  if(hot) { if(hot!=="auto" && (!/^\d+$/.test(hot)||!Number.isSafeInteger(Number(hot)))) throw new Error("GPU 预算需为 auto 或有效整数字节数"); set.hot_bytes=hot==="auto"?hot:Number(hot); } else set.hot_bytes=null;
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
  if(s.state==="stopped" && ["starting","ready","stopping"].includes(previousState))notice("模型已停止。",true);
  if(s.state==="ready" && previousState==="starting")notice("模型已就绪，可打开聊天。",true);
  const jobKey=String(j.started)+":"+j.state;
  if(j.state==="failed"&&jobKey!==lastJob)notice("准备失败。"+(j.error||j.message||"请查看下方准备日志。"));
  lastJob=jobKey;
  $("status").className="status "+s.state; $("status").querySelector("span").textContent=stateNames[s.state]||s.state;
  $("current-model").textContent=current.config.values.variant; $("endpoint").textContent=s.url;
  const progressText=phaseNames[j.phase]||phaseNames[j.state]||"尚无准备任务";
  $("progress-label").textContent=progressText+(j.file?" · "+j.file:"")+(j.error?" · "+j.error:"");
  $("progress").value=j.total?Math.min(100,100*j.bytes/j.total):j.state==="complete"?100:0;
  updateControls();
}
async function logs() { const value=await api("logs?kind="+logKind); const el=$("logs"),bottom=el.scrollTop+el.clientHeight>=el.scrollHeight-20; el.textContent=value.text||"暂无日志。";if(bottom)el.scrollTop=el.scrollHeight; }
async function action(fn) { if(busy)return;busy=true;updateControls();try{await fn();await refresh();await logs();}catch(e){notice(e.message);}finally{busy=false;updateControls();} }
$("config-form").addEventListener("input",edit);
$("variant").addEventListener("change",()=>{ if($("variant").value.startsWith("GSQ"))$("vision").checked=false; updateControls();notice("已切换版本，请点“按模型和目录填写路径”，或填写该版本已有文件的位置。"); });
$("config-form").addEventListener("submit",e=>{e.preventDefault();action(async()=>{const v=await api("config",payload());const state=await api("state");fill(state.config);notice("配置已保存。"+(v.backup?" 已保留上一版备份。":""),true);});});
$("use-default-paths").addEventListener("click",()=>action(async()=>{const v=payload();v.reset_paths=true;for(const key of ["model","draft","tokenizer","mmproj","engine","profile","heat","integrity"])delete v.set[key];await api("config",v);const state=await api("state");fill(state.config);notice("已按所选模型和目录保存默认路径。",true);}));
$("plan").addEventListener("click",()=>action(async()=>{if(dirty)throw new Error("先保存配置，再查看准备计划。");const p=await api("plan");const box=$("plan-view");box.replaceChildren();const title=document.createElement("strong");title.textContent=`待下载 ${(p.download_bytes/2**30).toFixed(2)} GiB · ${p.assets.length} 个文件`;box.append(title);const list=document.createElement("ul");for(const a of p.assets){const li=document.createElement("li");li.textContent=`${a.exists?(a.size_matches?"待校验":"大小不符，保留原文件"):a.partial_bytes?"续传":"下载"} · ${a.path} · ${(a.bytes/2**30).toFixed(2)} GiB`;if(a.exists&&!a.size_matches)li.className="bad";list.append(li);}for(const d of p.disks){const li=document.createElement("li");li.textContent=`${d.path}：可用 ${(d.free_bytes/2**30).toFixed(1)} GiB${d.enough?"":"，空间不足"}`;if(!d.enough)li.className="bad";list.append(li);}box.append(list);}));
$("doctor").addEventListener("click",()=>action(async()=>{const d=await api("doctor",{});notice(d.checks.map(c=>(c.ok?"✓ ":"✕ ")+c.name+"："+c.detail).join("\n"),d.ok);}));
for(const [id,launch]of [["prepare",false],["prepare-launch",true]])$(id).addEventListener("click",()=>action(async()=>{await api("install",{launch});logKind="install";selectLog();notice("准备任务已开始，可在下方查看进度与日志。",true);}));
$("start").addEventListener("click",()=>action(async()=>{await api("start",{});logKind="service";selectLog();notice("模型正在加载。就绪后可打开聊天。",true);}));
$("stop").addEventListener("click",()=>action(async()=>{await api("stop",{});notice("停止请求已发送，模型状态会自动更新。",true);}));
$("cancel-prepare").addEventListener("click",()=>action(async()=>{await api("cancel",{});notice("已取消准备，断点文件与日志保留。",true);}));
function selectLog(){ $("service-log").classList.toggle("selected",logKind==="service");$("install-log").classList.toggle("selected",logKind==="install"); }
for(const kind of ["service","install"])$(kind+"-log").addEventListener("click",()=>{logKind=kind;selectLog();logs().catch(e=>notice(e.message));});
$("refresh-log").addEventListener("click",()=>logs().catch(e=>notice(e.message)));
refresh().then(logs).catch(e=>notice(e.message));
setInterval(()=>{if(!busy)refresh().then(logs).catch(e=>notice("管理服务连接失败："+e.message));},2000);
