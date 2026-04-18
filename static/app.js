const $ = (id) => document.getElementById(id);
const state = { currentStream: null, startTs: 0, tokenCount: 0 };

function fmtBytes(n) {
  if (n == null || Number.isNaN(n)) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  let v = Number(n);
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 ? 0 : v >= 10 ? 1 : 2)} ${u[i]}`;
}

function fmtSeconds(s) {
  s = Math.floor(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  if (h) return `${h}h ${m}m`;
  if (m) return `${m}m ${sec}s`;
  return `${sec}s`;
}

function toast(msg, kind = "") {
  const t = $("toast");
  t.textContent = msg;
  t.className = `toast ${kind}`;
  setTimeout(() => t.classList.add("hidden"), 4000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    renderStatus(s);
  } catch (e) {
    // server probably not up yet
  }
}

function renderStatus(s) {
  const h = s.host || {};
  $("cpuMeter").value = h.cpu_percent ?? 0;
  $("cpuVal").textContent = `${(h.cpu_percent ?? 0).toFixed(0)}%`;
  $("memMeter").value = h.mem_percent ?? 0;
  $("memVal").textContent = `${fmtBytes(h.mem_used)} / ${fmtBytes(h.mem_total)}`;
  const mlx = h.mlx || {};
  $("mlxActive").textContent = mlx.active_bytes != null ? fmtBytes(mlx.active_bytes) : (mlx.error ? "n/a" : "—");
  $("mlxPeak").textContent = mlx.peak_bytes != null ? fmtBytes(mlx.peak_bytes) : "—";
  $("mlxCache").textContent = mlx.cache_bytes != null ? fmtBytes(mlx.cache_bytes) : "—";

  const m = s.model || {};
  const pill = $("enginePill");
  if (m.loading) { pill.textContent = `loading: ${m.loading}`; pill.className = "pill loading"; }
  else if (m.loaded) { pill.textContent = `loaded: ${m.name}`; pill.className = "pill loaded"; }
  else { pill.textContent = "engine: idle"; pill.className = "pill"; }

  const versions = s.versions || {};
  const vPill = $("versionsPill");
  if (vPill) {
    const vm = versions.mlx || "?";
    const vlm = versions.mlx_lm || "?";
    const py = versions.python ? ` · py ${versions.python}` : "";
    vPill.textContent = `mlx ${vm} · mlx-lm ${vlm}${py}`;
    vPill.title = Object.entries(versions).map(([k, v]) => `${k}: ${v}`).join("\n");
    vPill.classList.toggle("error", !!versions.python_too_old);
  }
  renderPythonWarning(versions);

  $("modelInfo").textContent = m.loaded
    ? `name:         ${m.name}
uptime:       ${fmtSeconds(m.uptime_seconds)}
generations:  ${m.generations}
total_tokens: ${m.total_tokens}`
    : "No model loaded.";

  renderEndpoint(m.loaded ? m.name : null);
  onModelStateChanged(m);

  const dl = $("modelList");
  if (s.suggested && dl.children.length === 0) {
    for (const name of s.suggested) {
      const opt = document.createElement("option");
      opt.value = name;
      dl.appendChild(opt);
    }
  }
}

$("loadBtn").addEventListener("click", async () => {
  const name = $("modelName").value.trim();
  if (!name) return toast("Enter a model name", "err");
  $("loadBtn").disabled = true;
  $("enginePill").textContent = `loading: ${name}`;
  $("enginePill").className = "pill loading";
  try {
    await api("/api/models/load", { method: "POST", body: JSON.stringify({ name }) });
    toast(`Loaded ${name}`, "ok");
  } catch (e) {
    toast(`Load failed: ${e.message}`, "err");
  } finally {
    $("loadBtn").disabled = false;
    refreshStatus();
  }
});

$("unloadBtn").addEventListener("click", async () => {
  try {
    await api("/api/models/unload", { method: "POST" });
    toast("Unloaded", "ok");
  } catch (e) {
    toast(`Unload failed: ${e.message}`, "err");
  } finally {
    refreshStatus();
  }
});

$("genBtn").addEventListener("click", async () => {
  const prompt = $("prompt").value;
  if (!prompt.trim()) return toast("Enter a prompt", "err");
  const body = {
    prompt,
    max_tokens: Number($("maxTokens").value),
    temperature: Number($("temperature").value),
    top_p: Number($("topP").value),
    system: $("system").value || null,
    stream: true,
  };
  const out = $("output");
  out.textContent = "";
  out.classList.remove("empty");
  $("genStats").textContent = "";
  $("genBtn").disabled = true;
  $("stopBtn").disabled = false;
  state.startTs = performance.now();
  state.tokenCount = 0;

  const controller = new AbortController();
  state.currentStream = controller;

  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    await consumeSSE(res, (event, obj) => {
      if (event === "error") throw new Error(obj.error || "stream error");
      if (event === "done") return;
      if (obj.delta) {
        out.textContent += obj.delta;
        state.tokenCount += 1;
        out.scrollTop = out.scrollHeight;
        updateStats();
      }
    });
  } catch (e) {
    if (e.name !== "AbortError") toast(`Generation failed: ${e.message}`, "err");
  } finally {
    $("genBtn").disabled = false;
    $("stopBtn").disabled = true;
    state.currentStream = null;
    updateStats(true);
    refreshStatus();
  }
});

$("stopBtn").addEventListener("click", () => {
  if (state.currentStream) state.currentStream.abort();
});

function updateStats(final = false) {
  const secs = (performance.now() - state.startTs) / 1000;
  const tps = secs > 0 ? state.tokenCount / secs : 0;
  $("genStats").textContent = `${state.tokenCount} chunks · ${secs.toFixed(1)}s · ${tps.toFixed(1)} chunks/s${final ? " (done)" : ""}`;
}

async function consumeSSE(res, onEvent) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, idx);
      buffer = buffer.slice(idx + 2);
      const lines = raw.split("\n");
      let event = "message";
      let data = "";
      for (const line of lines) {
        if (line.startsWith(":")) continue;
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      let obj;
      try { obj = JSON.parse(data); } catch { obj = { raw: data }; }
      onEvent(event, obj);
    }
  }
}

// ---- Hugging Face panel -----------------------------------------------

$("hfSearchBtn").addEventListener("click", hfSearch);
$("hfQuery").addEventListener("keydown", (e) => { if (e.key === "Enter") hfSearch(); });
$("hfRefreshCacheBtn").addEventListener("click", refreshCache);

async function hfSearch() {
  const q = $("hfQuery").value.trim();
  const author = $("hfAuthor").value.trim();
  const list = $("hfResults");
  list.innerHTML = `<li class="empty">Searching…</li>`;
  try {
    const params = new URLSearchParams({ q, author, limit: "30" });
    const r = await api(`/api/hf/search?${params}`);
    renderSearch(r.results || []);
  } catch (e) {
    list.innerHTML = `<li class="empty">Error: ${escapeHtml(e.message)}</li>`;
  }
}

function renderSearch(items) {
  const list = $("hfResults");
  if (!items.length) { list.innerHTML = `<li class="empty">No results</li>`; return; }
  list.innerHTML = "";
  for (const m of items) {
    const li = document.createElement("li");
    const dl = m.downloads != null ? `${formatCount(m.downloads)} dl` : "";
    const likes = m.likes ? ` · ♥ ${formatCount(m.likes)}` : "";
    const task = m.pipeline_tag ? ` · ${m.pipeline_tag}` : "";
    li.innerHTML = `
      <div class="name">${escapeHtml(m.id)}</div>
      <div class="meta">${dl}${likes}${task}</div>
      <div class="actions">
        <button data-act="download">Download</button>
        <button data-act="load">Load</button>
      </div>`;
    li.querySelector('[data-act="download"]').addEventListener("click", () => hfDownload(m.id));
    li.querySelector('[data-act="load"]').addEventListener("click", () => {
      $("modelName").value = m.id;
      $("loadBtn").click();
    });
    list.appendChild(li);
  }
}

async function hfDownload(repoId) {
  try {
    await api("/api/hf/download", { method: "POST", body: JSON.stringify({ name: repoId }) });
    toast(`Download started: ${repoId}`, "ok");
    refreshDownloads();
  } catch (e) {
    toast(`Download failed: ${e.message}`, "err");
  }
}

async function refreshDownloads() {
  try {
    const r = await api("/api/hf/downloads");
    renderDownloads(r.jobs || []);
  } catch {}
}

function renderDownloads(jobs) {
  const list = $("hfDownloads");
  if (!jobs.length) { list.innerHTML = `<li class="empty">No downloads yet</li>`; return; }
  list.innerHTML = "";
  jobs.sort((a, b) => (b.started_at || 0) - (a.started_at || 0));
  for (const j of jobs) {
    const li = document.createElement("li");
    li.className = "job";
    const pct = j.percent != null ? j.percent : (j.total_bytes ? (j.bytes_downloaded / j.total_bytes) * 100 : 0);
    const badge = j.status === "done" ? "ok" : j.status === "error" ? "err" : "warn";
    const size = `${fmtBytes(j.bytes_downloaded)}${j.total_bytes ? " / " + fmtBytes(j.total_bytes) : ""}`;
    li.innerHTML = `
      <div class="name">${escapeHtml(j.repo_id)}<span class="badge ${badge}">${j.status}</span></div>
      <div class="meta">${size} · ${j.files_done}/${j.files_total || "?"} files${j.error ? " · " + escapeHtml(j.error) : ""}</div>
      <div class="actions">
        <button data-act="load">Load</button>
      </div>
      <div class="progress"><div style="width:${Math.max(0, Math.min(100, pct)).toFixed(1)}%"></div></div>
    `;
    li.querySelector('[data-act="load"]').addEventListener("click", () => {
      $("modelName").value = j.repo_id;
      $("loadBtn").click();
    });
    list.appendChild(li);
  }
}

async function refreshCache() {
  const list = $("hfCache");
  list.innerHTML = `<li class="empty">Scanning…</li>`;
  try {
    const r = await api("/api/hf/cache");
    $("hfCacheTotal").textContent = `Total on disk: ${fmtBytes(r.size_on_disk || 0)} · ${(r.repos || []).length} repos`;
    renderCache(r.repos || []);
  } catch (e) {
    list.innerHTML = `<li class="empty">Error: ${escapeHtml(e.message)}</li>`;
  }
}

function renderCache(repos) {
  const list = $("hfCache");
  if (!repos.length) { list.innerHTML = `<li class="empty">Cache is empty</li>`; return; }
  list.innerHTML = "";
  for (const repo of repos) {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="name">${escapeHtml(repo.repo_id)}</div>
      <div class="meta">${fmtBytes(repo.size_on_disk)} · ${repo.nb_files} files</div>
      <div class="actions">
        <button data-act="load">Load</button>
        <button data-act="delete">Delete</button>
      </div>`;
    li.querySelector('[data-act="load"]').addEventListener("click", () => {
      $("modelName").value = repo.repo_id;
      $("loadBtn").click();
    });
    li.querySelector('[data-act="delete"]').addEventListener("click", async () => {
      if (!confirm(`Delete ${repo.repo_id} from local cache?`)) return;
      try {
        const res = await api("/api/hf/delete", { method: "POST", body: JSON.stringify({ name: repo.repo_id }) });
        toast(`Freed ${fmtBytes(res.freed_bytes || 0)}`, "ok");
        refreshCache();
      } catch (e) {
        toast(`Delete failed: ${e.message}`, "err");
      }
    });
    list.appendChild(li);
  }
}

function formatCount(n) {
  if (n == null) return "";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---- Model Settings ---------------------------------------------------

let settingsLoadedFor = null;

function onModelStateChanged(m) {
  const wrap = $("modelSettingsWrap");
  if (!wrap) return;
  if (!m.loaded) {
    wrap.style.display = "none";
    settingsLoadedFor = null;
    return;
  }
  wrap.style.display = "";
  if (settingsLoadedFor !== m.name) {
    loadModelSettings(m.name);
  }
}

async function loadModelSettings(name) {
  settingsLoadedFor = name;
  $("settingsHint").textContent = "Loading…";
  try {
    const r = await api(`/api/settings/models/${encodeURI(name)}`);
    const s = r.settings || {};
    $("setSystem").value = s.system || "";
    $("setTemperature").value = s.temperature != null ? s.temperature : "";
    $("setTopP").value = s.top_p != null ? s.top_p : "";
    $("setMaxTokens").value = s.max_tokens != null ? s.max_tokens : "";
    $("setAutoload").checked = !!s.autoload;
    // default on if unset
    $("setStripThinking").checked = s.strip_thinking !== false;
    $("setEnableThinking").value =
      s.enable_thinking === true ? "true" :
      s.enable_thinking === false ? "false" : "auto";
    $("settingsHint").textContent = Object.keys(s).length ? "Loaded." : "No overrides — using defaults.";
  } catch (e) {
    $("settingsHint").textContent = `Load failed: ${e.message}`;
  }
}

$("saveSettingsBtn").addEventListener("click", async () => {
  if (!settingsLoadedFor) return toast("No model loaded", "err");
  const body = {
    system: $("setSystem").value.trim() || null,
    temperature: $("setTemperature").value === "" ? null : Number($("setTemperature").value),
    top_p: $("setTopP").value === "" ? null : Number($("setTopP").value),
    max_tokens: $("setMaxTokens").value === "" ? null : Number($("setMaxTokens").value),
    autoload: $("setAutoload").checked,
    strip_thinking: $("setStripThinking").checked,
    enable_thinking:
      $("setEnableThinking").value === "true" ? true :
      $("setEnableThinking").value === "false" ? false : null,
  };
  try {
    await api(`/api/settings/models/${encodeURI(settingsLoadedFor)}`, {
      method: "PUT",
      body: JSON.stringify(body),
    });
    $("settingsHint").textContent = "Saved.";
    toast("Settings saved", "ok");
  } catch (e) {
    toast(`Save failed: ${e.message}`, "err");
  }
});

$("resetSettingsBtn").addEventListener("click", async () => {
  if (!settingsLoadedFor) return;
  if (!confirm(`Clear saved settings for ${settingsLoadedFor}?`)) return;
  try {
    await api(`/api/settings/models/${encodeURI(settingsLoadedFor)}`, { method: "DELETE" });
    await loadModelSettings(settingsLoadedFor);
    toast("Settings cleared", "ok");
  } catch (e) {
    toast(`Reset failed: ${e.message}`, "err");
  }
});

// ---- API Endpoint info ------------------------------------------------

function renderEndpoint(modelName) {
  const base = `${location.origin}/v1`;
  const chat = `${base}/chat/completions`;
  $("apiBaseUrl").textContent = base;
  $("apiChatUrl").textContent = chat;
  $("apiModelId").textContent = modelName || "(no model loaded)";

  const badge = $("apiStatusBadge");
  if (modelName) {
    badge.textContent = "ready";
    badge.className = "badge ok";
  } else {
    badge.textContent = "no model loaded";
    badge.className = "badge warn";
  }

  const name = modelName || "mlx-community/Llama-3.2-1B-Instruct-4bit";
  const shortName = name.split("/").pop() + " (MLX)";

  // ── OpenCode ──────────────────────────────────────────────────────────────
  $("opencodeExample").textContent = JSON.stringify({
    $schema: "https://opencode.ai/config.json",
    provider: {
      mlxr: {
        npm: "@ai-sdk/openai-compatible",
        name: "MLXr (local)",
        options: { baseURL: base, apiKey: "not-needed" },
        models: { [name]: { name: shortName } },
      },
    },
  }, null, 2);

  // ── Zed ───────────────────────────────────────────────────────────────────
  // Merge into existing ~/.config/zed/settings.json; these are the relevant keys.
  $("zedExample").textContent = JSON.stringify({
    language_models: {
      openai: {
        api_url: base,
        available_models: [
          { name, display_name: shortName, max_tokens: 32768 },
        ],
      },
    },
    assistant: {
      default_model: { provider: "openai", model: name },
      version: "2",
    },
  }, null, 2);

  // ── Cursor ────────────────────────────────────────────────────────────────
  // Settings → Features → OpenAI API Key section → override base URL.
  $("cursorExample").textContent =
`# Cursor → Settings → Features → OpenAI API Key
#   API Key:  not-needed
#   Override OpenAI Base URL:  ${base}

# Or in .cursor/mcp.json for project-level:
${JSON.stringify({ openai: { baseUrl: base, apiKey: "not-needed" } }, null, 2)}`;

  // ── Continue (VS Code / JetBrains) ────────────────────────────────────────
  $("continueExample").textContent = JSON.stringify({
    models: [
      {
        title: shortName,
        provider: "openai",
        model: name,
        apiBase: base,
        apiKey: "not-needed",
      },
    ],
  }, null, 2);

  // ── Aider ─────────────────────────────────────────────────────────────────
  $("aiderExample").textContent =
`# One-off:
aider \\
  --openai-api-base ${base} \\
  --openai-api-key not-needed \\
  --model openai/${name}

# Or persist in ~/.aider.conf.yml:
openai-api-base: ${base}
openai-api-key: not-needed
model: openai/${name}`;

  // ── Neovim / avante.nvim ──────────────────────────────────────────────────
  $("neoExample").textContent =
`-- ~/.config/nvim/init.lua  (or your lazy.nvim spec)
require("avante").setup({
  provider = "openai",
  openai = {
    endpoint = "${base}",
    model    = "${name}",
    api_key  = "not-needed",
    max_tokens = 32768,
  },
})`;

  // ── curl ──────────────────────────────────────────────────────────────────
  $("curlExample").textContent =
`curl ${chat} \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer not-needed" \\
  -d '${JSON.stringify({ model: name, messages: [{ role: "user", content: "Hello!" }], stream: false })}'`;

  // ── OpenAI Python SDK ─────────────────────────────────────────────────────
  $("pythonExample").textContent =
`from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="not-needed")
resp = client.chat.completions.create(
    model="${name}",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)`;
}

// Delegated copy handler for any button with data-copy-target
document.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-copy-target]");
  if (!btn) return;
  const target = $(btn.dataset.copyTarget);
  if (!target) return;
  const text = target.textContent || "";
  navigator.clipboard.writeText(text).then(
    () => {
      const prev = btn.textContent;
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = prev), 1200);
    },
    () => toast("Copy failed", "err"),
  );
});

function renderPythonWarning(versions) {
  const el = $("pythonWarn");
  if (!el) return;
  if (!versions || !versions.python_too_old) {
    el.classList.add("hidden");
    el.innerHTML = "";
    return;
  }
  const py = versions.python || "?";
  const min = versions.python_min || "3.10";
  el.classList.remove("hidden");
  el.innerHTML = `
    <strong>Python ${escapeHtml(py)} is too old.</strong>
    Current mlx-lm requires Python ≥ ${escapeHtml(min)}.
    Upgrading packages in this venv will only get you a stale mlx-lm that lacks newer architectures (e.g. <code>qwen3_5_moe</code>).
    <br/>Recreate the venv:
    <br/><code>rm -rf .venv &amp;&amp; python3.12 -m venv .venv &amp;&amp; ./run.sh</code>
  `;
}

// ---- Engine self-update ------------------------------------------------

let lastCheck = null;

$("checkUpdatesBtn").addEventListener("click", checkUpdates);
$("upgradeBtn").addEventListener("click", runUpgrade);
$("restartBtn").addEventListener("click", restartServer);

async function checkUpdates() {
  $("checkUpdatesBtn").disabled = true;
  $("engineHint").textContent = "Checking PyPI…";
  try {
    const r = await api("/api/engine/check");
    lastCheck = r;
    renderEngineVersions(r.packages);
    // If Python is too old, upgrading in-place won't actually move mlx-lm forward.
    const pythonTooOld = r.python_too_old === true;
    $("upgradeBtn").disabled = !r.update_available || pythonTooOld;
    $("engineHint").textContent = pythonTooOld
      ? "Python is too old — recreate .venv before upgrading."
      : r.update_available
        ? "Updates available."
        : "All up to date.";
  } catch (e) {
    $("engineHint").textContent = `Check failed: ${e.message}`;
  } finally {
    $("checkUpdatesBtn").disabled = false;
  }
}

function renderEngineVersions(pkgs) {
  const table = $("engineVersions");
  table.innerHTML = "";
  for (const [name, info] of Object.entries(pkgs)) {
    const tr = document.createElement("tr");
    const status = info.update_available
      ? `<span class="badge warn">update: ${escapeHtml(info.latest)}</span>`
      : (info.installed === info.latest ? `<span class="badge ok">latest</span>` : `<span class="badge">—</span>`);
    tr.innerHTML = `
      <td class="pkg">${escapeHtml(name)}</td>
      <td class="ver">${escapeHtml(info.installed)}${info.latest ? " · latest " + escapeHtml(info.latest) : ""}</td>
      <td class="status">${status}</td>`;
    table.appendChild(tr);
  }
}

async function runUpgrade() {
  if (!confirm("Run pip install --upgrade for mlx, mlx-lm, huggingface_hub, transformers in the server's venv?")) return;
  const log = $("engineLog");
  log.classList.remove("hidden");
  log.textContent = "";
  $("upgradeBtn").disabled = true;
  $("checkUpdatesBtn").disabled = true;
  $("restartBtn").disabled = true;

  let rc = null;
  try {
    const res = await fetch("/api/engine/upgrade", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ packages: ["mlx", "mlx-lm", "huggingface_hub", "transformers"] }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    await consumeSSE(res, (event, obj) => {
      if (event === "start") {
        log.textContent += `$ ${obj.cmd.join(" ")}\n\n`;
      } else if (event === "done") {
        rc = obj.returncode;
        log.textContent += `\n--- exit ${rc} ---\n`;
      } else if (event === "error") {
        log.textContent += `\nERROR: ${obj.error || "stream error"}\n`;
      } else if (obj.line != null) {
        log.textContent += obj.line + "\n";
        log.scrollTop = log.scrollHeight;
      }
    });
    if (rc === 0) {
      toast("Upgrade complete — restart the server to apply.", "ok");
      $("engineHint").textContent = "Upgrade done. Click Restart to load the new versions.";
    } else {
      toast(`pip exited with ${rc}`, "err");
    }
  } catch (e) {
    toast(`Upgrade failed: ${e.message}`, "err");
    log.textContent += `\nFAILED: ${e.message}\n`;
  } finally {
    $("checkUpdatesBtn").disabled = false;
    $("restartBtn").disabled = false;
    checkUpdates();
  }
}

async function restartServer() {
  if (!confirm("Restart the MLXr server? Any loaded model will be unloaded.")) return;
  $("restartBtn").disabled = true;
  $("engineHint").textContent = "Restarting…";
  try {
    const r = await api("/api/engine/restart", { method: "POST" });
    if (!r.managed_by_run_sh) {
      toast("Server exited. If not launched via run.sh, rerun manually.", "err");
    }
  } catch {
    // connection will drop — that's expected
  }
  waitForServer().then(() => {
    $("engineHint").textContent = "Back online.";
    $("restartBtn").disabled = false;
    refreshStatus();
    checkUpdates();
  });
}

async function waitForServer(maxWaitMs = 30000) {
  const start = Date.now();
  while (Date.now() - start < maxWaitMs) {
    try {
      const res = await fetch("/api/engine/version", { cache: "no-store" });
      if (res.ok) return true;
    } catch {}
    await new Promise((r) => setTimeout(r, 750));
  }
  return false;
}

$("output").classList.add("empty");
// Populate endpoint info up-front so Base URL / examples are always visible,
// even if the backend hasn't responded yet (e.g. in a preview panel).
renderEndpoint(null);
refreshStatus();
refreshCache();
refreshDownloads();
checkUpdates();
setInterval(refreshStatus, 2500);
setInterval(refreshDownloads, 1500);
