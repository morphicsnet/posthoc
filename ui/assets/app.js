(function(){
  const view = document.getElementById("view");
  const navLinks = Array.from(document.querySelectorAll("nav a"));

  function setActive(key){
    navLinks.forEach(a => a.classList.toggle("active", a.dataset.view === key));
  }

  async function api(path, opts={}){
    const r = await fetch(path, opts);
    const ct = r.headers.get("content-type") || "";
    if (ct.includes("application/json")) return r.json();
    return r.text();
  }

  async function showCommandCenter(){
    setActive("cmd");
    const health = await api("/healthz").catch(()=>({ok:false}));
    view.innerHTML = `
      <div class="grid">
        <div class="card"><h3>Health</h3><div class="mono">${escapeHtml(JSON.stringify(health))}</div></div>
        <div class="card"><h3>Latest Test Report</h3>
          <button id="btn-json">Download JSON</button>
          <button id="btn-md">Download MD</button>
          <div class="mono" id="reportNote"></div>
        </div>
        <div class="card"><h3>Metrics Aggregate</h3>
          <form id="fm-metrics">
            <label>Endpoints (newline separated)</label><br/>
            <textarea id="metrics-endpoints" rows="4" cols="50">http://localhost:8080/metrics
http://localhost:9090/metrics</textarea><br/>
            <button class="primary" type="submit">Aggregate</button>
          </form>
          <div class="mono" id="metrics-out"></div>
        </div>
      </div>`;
    document.getElementById("btn-json").onclick = ()=> window.open("/api/reports/export?format=json","_blank");
    document.getElementById("btn-md").onclick = ()=> window.open("/api/reports/export?format=md","_blank");
    document.getElementById("fm-metrics").onsubmit = async (e)=>{
      e.preventDefault();
      const endpoints = document.getElementById("metrics-endpoints").value.split(/\\n+/).map(s=>s.trim()).filter(Boolean);
      const out = await api("/api/metrics/aggregate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({endpoints})})
                      .catch(err=>({error:String(err)}));
      document.getElementById("metrics-out").textContent = JSON.stringify(out, null, 2);
    };
  }

  async function showTests(){
    setActive("tests");
    view.innerHTML = `
      <div class="card">
        <h3>Start E2E Test Run</h3>
        <form id="fm-run">
          <label>Base URL</label><br/><input id="arg-base" value="http://localhost:8080"/><br/>
          <label>Metrics (Gateway)</label><br/><input id="arg-mg" value="http://localhost:8080/metrics"/><br/>
          <label>Metrics (Explainer)</label><br/><input id="arg-me" value="http://localhost:9090/metrics"/><br/>
          <label>Duration (s)</label><br/><input id="arg-dur" value="60" type="number"/><br/>
          <label>Concurrency</label><br/><input id="arg-conc" value="100" type="number"/><br/>
          <button class="primary" type="submit">Start</button>
        </form>
      </div>
      <div class="card"><h3>Recent Runs</h3><div id="runs" class="mono"></div></div>
    `;
    document.getElementById("fm-run").onsubmit = async (e)=>{
      e.preventDefault();
      const body = {
        base_url: document.getElementById("arg-base").value,
        metrics_gateway: document.getElementById("arg-mg").value,
        metrics_explainer: document.getElementById("arg-me").value,
        load_duration: parseInt(document.getElementById("arg-dur").value,10),
        concurrency: parseInt(document.getElementById("arg-conc").value,10),
      };
      const tr = await api("/api/testruns", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)})
                   .catch(err=>({error:String(err)}));
      renderRuns(); 
      if (tr && tr.id){ streamRun(tr.id); }
    };
    renderRuns();
  }

  async function renderRuns(){
    const runsEl = document.getElementById("runs");
    const runs = await api("/api/testruns").catch(()=>[]);
    runsEl.textContent = JSON.stringify(runs, null, 2);
  }

  function streamRun(id){
    let proto = (location.protocol === "https:") ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/testruns/${id}/stream`);
    ws.onmessage = (ev)=> {
      try{
        const obj = JSON.parse(ev.data);
        if (obj.type === "testrun.update"){
          renderRuns();
        }
      }catch(e){}
    };
    ws.onclose = ()=>{};
  }

  async function showObservability(){
    setActive("observability");
    view.innerHTML = `
      <div class="card"><h3>Aggregate Metrics</h3>
        <form id="fm-m">
          <label>Metrics endpoints</label><br/>
          <textarea id="m-list" rows="4" cols="50">http://localhost:8080/metrics
http://localhost:9090/metrics</textarea><br/>
          <button class="primary" type="submit">Query</button>
        </form>
        <div class="mono" id="m-out"></div>
      </div>
    `;
    document.getElementById("fm-m").onsubmit = async (e)=>{
      e.preventDefault();
      const endpoints = document.getElementById("m-list").value.split(/\\n+/).map(s=>s.trim()).filter(Boolean);
      const out = await api("/api/metrics/aggregate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({endpoints})})
                      .catch(err=>({error:String(err)}));
      document.getElementById("m-out").textContent = JSON.stringify(out, null, 2);
    };
  }

  function showDeployments(){
    setActive("deployments");
    view.innerHTML = `<div class="card"><h3>Helm Checks</h3><button id="btn-checks" class="primary">Run checks</button><pre class="mono" id="out"></pre></div>`;
    document.getElementById("btn-checks").onclick = async ()=>{
      const out = await api("/api/helm/checks").catch(err=>({error:String(err)}));
      document.getElementById("out").textContent = JSON.stringify(out, null, 2);
    };
  }

  function showAutoscaling(){
    setActive("autoscaling");
    view.innerHTML = `
      <div class="grid">
        <div class="card"><h3>KEDA</h3><button id="btn-keda" class="primary">Load</button><pre class="mono" id="keda"></pre></div>
        <div class="card"><h3>Karpenter</h3><button id="btn-karp" class="primary">Load</button><pre class="mono" id="karp"></pre></div>
      </div>`;
    document.getElementById("btn-keda").onclick = async ()=>{
      const out = await api("/api/keda/scalers").catch(err=>({error:String(err)}));
      document.getElementById("keda").textContent = JSON.stringify(out, null, 2);
    };
    document.getElementById("btn-karp").onclick = async ()=>{
      const out = await api("/api/karpenter/nodepools").catch(err=>({error:String(err)}));
      document.getElementById("karp").textContent = JSON.stringify(out, null, 2);
    };
  }

  function showSecurity(){
    setActive("security");
    view.innerHTML = `
      <div class="card"><h3>Audit Search</h3>
        <form id="fm-a">
          <label>Query</label><br/><input id="q" placeholder="chat.submit"/><br/>
          <label>Path</label><br/><input id="p" value="/var/log/hypergraph/audit.log"/><br/>
          <button class="primary" type="submit">Search</button>
        </form>
        <pre class="mono" id="a-out"></pre>
      </div>`;
    document.getElementById("fm-a").onsubmit = async (e)=>{
      e.preventDefault();
      const q = document.getElementById("q").value;
      const p = document.getElementById("p").value;
      const out = await api(`/api/audit/search?q=${encodeURIComponent(q)}&path=${encodeURIComponent(p)}`).catch(err=>({error:String(err)}));
      document.getElementById("a-out").textContent = JSON.stringify(out, null, 2);
    };
  }

  function showCompliance(){
    setActive("compliance");
    view.innerHTML = `
      <div class="grid">
        <div class="card"><h3>OpenAPI /v1</h3><button id="btn-openapi" class="primary">Check</button><pre class="mono" id="openapi"></pre></div>
        <div class="card"><h3>HIF Validate</h3>
          <textarea id="hif" rows="6" cols="50">{ "meta": { "version": "hif-1" } }</textarea><br/>
          <button id="btn-hif" class="primary">Validate</button>
          <pre class="mono" id="hif-out"></pre>
        </div>
      </div>`;
    document.getElementById("btn-openapi").onclick = async ()=>{
      const out = await api("/api/compliance/openapi/check").catch(err=>({error:String(err)}));
      document.getElementById("openapi").textContent = JSON.stringify(out, null, 2);
    };
    document.getElementById("btn-hif").onclick = async ()=>{
      let payload = {};
      try{ payload = JSON.parse(document.getElementById("hif").value) }catch(e){}
      const out = await api("/api/compliance/hif/validate", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({payload})})
                    .catch(err=>({error:String(err)}));
      document.getElementById("hif-out").textContent = JSON.stringify(out, null, 2);
    };
  }

  function showReports(){
    setActive("reports");
    view.innerHTML = `
      <div class="card"><h3>Export</h3>
        <button id="dl-json" class="primary">Download JSON</button>
        <button id="dl-md" class="primary">Download MD</button>
      </div>`;
    document.getElementById("dl-json").onclick = ()=> window.open("/api/reports/export?format=json","_blank");
    document.getElementById("dl-md").onclick = ()=> window.open("/api/reports/export?format=md","_blank");
  }

  function showSettings(){
    setActive("settings");
    view.innerHTML = `
      <div class="card">
        <h3>Session</h3>
        <p>Configure tokens via environment (AUTH_MODE=static, AUTH_TOKENS_JSON). Control plane aligns with [rbac_dependency()](services/gateway/src/rbac.py:62).</p>
        <p>Static UI; for production use a proper SPA build.</p>
      </div>`;
  }

  function onNavClick(ev){
    ev.preventDefault();
    const key = ev.target.dataset.view;
    if (key === "cmd") return showCommandCenter();
    if (key === "tests") return showTests();
    if (key === "observability") return showObservability();
    if (key === "deployments") return showDeployments();
    if (key === "autoscaling") return showAutoscaling();
    if (key === "security") return showSecurity();
    if (key === "compliance") return showCompliance();
    if (key === "reports") return showReports();
    if (key === "settings") return showSettings();
  }

  function escapeHtml(s){ return String(s).replace(/[&<>"]/g, m => ({'&':'&','<':'<','>':'>','"':'"'}[m])); }

  document.querySelectorAll("nav a").forEach(a => a.addEventListener("click", onNavClick));
  showCommandCenter();
})(); 