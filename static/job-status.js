(function () {
  const panel = document.querySelector("[data-job-panel]");
  if (!panel) return;

  const statusUrl = panel.dataset.statusUrl;
  const pollInterval = Number(panel.dataset.pollInterval || "2000");
  const titleEl = panel.querySelector('[data-field="title"]');
  const statusEl = panel.querySelector('[data-field="status"]');
  const stageEl = panel.querySelector('[data-field="progress-stage"]');
  const messageEl = panel.querySelector('[data-field="status-message"]');
  const artifactsEl = panel.querySelector('[data-field="artifact-links"]');
  const previewBoxEl = panel.querySelector('[data-field="preview-box"]');
  const previewEl = panel.querySelector('[data-field="preview"]');
  const errorEl = panel.querySelector('[data-field="error"]');
  const errorMessageEl = panel.querySelector('[data-field="error-message"]');
  const errorDetailBoxEl = panel.querySelector('[data-field="error-detail-box"]');
  const errorDetailEl = panel.querySelector('[data-field="error-detail"]');

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function isTerminal(status) {
    return status === "completed" || status === "failed";
  }

  function renderArtifacts(links) {
    if (!artifactsEl) return;
    if (!links || !links.length) {
      artifactsEl.innerHTML = '<li class="muted">Artifacts will appear here when the job completes.</li>';
      return;
    }

    artifactsEl.innerHTML = links
      .map((link) => {
        const copyButton = link.copyable
          ? `<button type="button" class="ghost" data-copy-url="${escapeHtml(link.href)}">Copy text</button>`
          : "";
        return `<li><a href="${escapeHtml(link.href)}">${escapeHtml(link.label)}</a>${copyButton}</li>`;
      })
      .join("");
  }

  async function copyText(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const text = await response.text();
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }

    const shadow = document.createElement("textarea");
    shadow.value = text;
    shadow.style.position = "fixed";
    shadow.style.left = "-9999px";
    document.body.appendChild(shadow);
    shadow.focus();
    shadow.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(shadow);
    if (!ok) throw new Error("copy failed");
  }

  if (artifactsEl) {
    artifactsEl.addEventListener("click", async (event) => {
      const button = event.target.closest("[data-copy-url]");
      if (!button) return;
      event.preventDefault();
      button.disabled = true;
      try {
        await copyText(button.dataset.copyUrl);
        button.textContent = "Copied";
        window.setTimeout(() => {
          button.textContent = "Copy text";
        }, 1500);
      } catch (error) {
        console.error(error);
        button.textContent = "Copy failed";
        window.setTimeout(() => {
          button.textContent = "Copy text";
        }, 1500);
      } finally {
        button.disabled = false;
      }
    });
  }

  function update(payload) {
    if (titleEl && payload.display_title) titleEl.textContent = payload.display_title;
    if (statusEl) {
      statusEl.textContent = payload.status;
      statusEl.className = `badge badge-${payload.status}`;
    }
    if (stageEl) stageEl.textContent = payload.progress_stage;
    if (messageEl) messageEl.textContent = payload.status_message;
    if (previewEl && payload.preview) {
      previewEl.textContent = payload.preview;
      if (previewBoxEl) previewBoxEl.hidden = false;
    }
    if (errorEl) {
      errorEl.hidden = payload.status !== "failed";
      if (errorMessageEl) errorMessageEl.textContent = payload.last_error_message || "";
      if (errorDetailEl) errorDetailEl.textContent = payload.last_error_detail || "";
      if (errorDetailBoxEl) errorDetailBoxEl.hidden = !payload.last_error_detail;
    }
    renderArtifacts(payload.artifact_links);
  }

  async function poll() {
    const response = await fetch(statusUrl, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) return;

    const payload = await response.json();
    update(payload);
    if (!isTerminal(payload.status)) {
      window.setTimeout(poll, pollInterval);
    }
  }

  poll().catch((error) => {
    console.error(error);
  });
})();
