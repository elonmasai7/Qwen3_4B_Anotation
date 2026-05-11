(function () {
  "use strict";

  const API_BASE = "/api/v1";

  function getEl(id) {
    return document.getElementById(id);
  }

  function show(el) {
    if (el) el.hidden = false;
  }

  function hide(el) {
    if (el) el.hidden = true;
  }

  function setText(el, text) {
    if (el) el.textContent = text;
  }

  function setHTML(el, html) {
    if (el) el.innerHTML = html;
  }

  function createStatusBadge(status) {
    const span = document.createElement("span");
    span.className = "status-badge status-" + status.toLowerCase();
    span.textContent = status;
    return span;
  }

  function formatNumber(n) {
    if (n == null) return "--";
    return Number(n).toLocaleString(undefined, {
      maximumFractionDigits: 3,
    });
  }

  function loadingBtn(btn, isLoading) {
    if (!btn) return;
    if (isLoading) {
      btn.classList.add("loading");
      btn.disabled = true;
    } else {
      btn.classList.remove("loading");
      btn.disabled = false;
    }
  }

  function showAlert(container, message, type) {
    if (!container) return;
    const div = document.createElement("div");
    div.className = "alert alert-" + type;
    div.textContent = message;
    container.appendChild(div);
    setTimeout(function () {
      div.remove();
    }, 5000);
  }

  async function apiFetch(path, options) {
    const url = API_BASE + path;
    const res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(function () {
        return { detail: res.statusText };
      });
      throw new Error(err.detail || "Request failed");
    }
    return res.json();
  }

  /* ---- Navigation ---- */
  var navToggle = document.querySelector(".nav-toggle");
  var navList = document.getElementById("nav-list");

  if (navToggle && navList) {
    navToggle.addEventListener("click", function () {
      var expanded = navToggle.getAttribute("aria-expanded") === "true";
      navToggle.setAttribute("aria-expanded", !expanded);
      navList.classList.toggle("open");
    });

    document.addEventListener("click", function (e) {
      if (!navToggle.contains(e.target) && !navList.contains(e.target)) {
        navToggle.setAttribute("aria-expanded", "false");
        navList.classList.remove("open");
      }
    });
  }

  /* ---- Dashboard ---- */
  function loadDashboard() {
    var statDatasets = getEl("stat-datasets");
    var statExperiments = getEl("stat-experiments");
    var statAnnotations = getEl("stat-annotations");
    var statPrompts = getEl("stat-prompts");

    if (!statDatasets) return;

    Promise.all([
      apiFetch("/datasets"),
      apiFetch("/experiments"),
      apiFetch("/prompts"),
    ])
      .then(function (results) {
        var datasets = results[0];
        var experiments = results[1];
        var prompts = results[2];
        setText(statDatasets, (datasets.datasets || []).length);
        setText(statExperiments, (experiments.experiments || []).length);
        setText(statPrompts, (prompts.prompts || []).length);

        var totalAnnotations = (experiments.experiments || []).reduce(function (
          acc,
          exp
        ) {
          return acc + (exp.annotation_count || 0);
        }, 0);
        setText(statAnnotations, totalAnnotations || 0);
      })
      .catch(function () {
        setText(statDatasets, "--");
        setText(statExperiments, "--");
        setText(statAnnotations, "--");
        setText(statPrompts, "--");
      });

    apiFetch("/experiments")
      .then(function (data) {
        var tbody = document.querySelector("#recent-experiments tbody");
        if (!tbody) return;
        var exps = data.experiments || [];
        if (exps.length === 0) {
          tbody.innerHTML =
            '<tr><td colspan="4" class="empty-state">No experiments yet.</td></tr>';
          return;
        }
        tbody.innerHTML = exps
          .slice(0, 5)
          .map(function (exp) {
            return (
              "<tr>" +
              "<td>" +
              escapeHTML(exp.name || "Unnamed") +
              "</td>" +
              "<td>" +
              createStatusBadge(exp.status || "unknown").outerHTML +
              "</td>" +
              "<td>" +
              formatNumber(exp.metrics?.accuracy) +
              "</td>" +
              "<td>" +
              (exp.created_at
                ? new Date(exp.created_at).toLocaleDateString()
                : "--") +
              "</td>" +
              "</tr>"
            );
          })
          .join("");
      })
      .catch(function () {});

    apiFetch("/leaderboard")
      .then(function (data) {
        var tbody = document.querySelector("#leaderboard-preview tbody");
        if (!tbody) return;
        var rankings = data.rankings || [];
        if (rankings.length === 0) {
          tbody.innerHTML =
            '<tr><td colspan="3" class="empty-state">No entries yet.</td></tr>';
          return;
        }
        tbody.innerHTML = rankings
          .slice(0, 3)
          .map(function (entry) {
            return (
              "<tr>" +
              "<td>" +
              entry.rank +
              "</td>" +
              "<td>" +
              escapeHTML(entry.name || "Unnamed") +
              "</td>" +
              "<td>" +
              formatNumber(entry.score) +
              "</td>" +
              "</tr>"
            );
          })
          .join("");
      })
      .catch(function () {});
  }

  function escapeHTML(str) {
    var div = document.createElement("div");
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
  }

  /* ---- Annotation ---- */
  function setupAnnotation() {
    var form = getEl("annotation-form");
    if (!form) return;

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var btn = getEl("annotate-btn");
      var resultSection = getEl("annotation-result");
      var resultId = getEl("result-id");
      var resultStatus = getEl("result-status");
      var resultLabels = getEl("result-labels");
      var resultTime = getEl("result-time");

      var data = {
        data_id: getEl("data-id").value.trim(),
        content: getEl("content").value.trim(),
        prompt_template: getEl("prompt-template").value,
      };

      if (!data.data_id || !data.content || !data.prompt_template) {
        return;
      }

      loadingBtn(btn, true);
      hide(resultSection);

      apiFetch("/annotate", {
        method: "POST",
        body: JSON.stringify(data),
      })
        .then(function (res) {
          setText(resultId, res.annotation_id);
          setText(resultStatus, res.status);
          setText(
            resultLabels,
            (res.labels || [])
              .map(function (l) {
                return l.label || l.value || JSON.stringify(l);
              })
              .join(", ") || "none"
          );
          setText(
            resultTime,
            res.processing_time_ms != null
              ? res.processing_time_ms + " ms"
              : "--"
          );
          show(resultSection);
        })
        .catch(function (err) {
          setText(resultId, "Error");
          setText(resultStatus, err.message);
          show(resultSection);
        })
        .finally(function () {
          loadingBtn(btn, false);
        });
    });
  }

  /* ---- Datasets ---- */
  function setupDatasets() {
    var form = getEl("upload-form");
    if (!form) return;

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var btn = getEl("upload-btn");
      var fileInput = getEl("file-upload");
      var status = getEl("upload-status");

      if (!fileInput.files.length) return;

      loadingBtn(btn, true);
      var fd = new FormData();
      fd.append("file", fileInput.files[0]);

      fetch(API_BASE + "/datasets/upload", {
        method: "POST",
        body: fd,
      })
        .then(function (r) {
          if (!r.ok) throw new Error("Upload failed");
          return r.json();
        })
        .then(function (res) {
          showAlert(status, "Uploaded: " + res.dataset_id, "success");
          loadDatasetsTable();
          fileInput.value = "";
        })
        .catch(function (err) {
          showAlert(status, err.message, "error");
        })
        .finally(function () {
          loadingBtn(btn, false);
        });
    });

    loadDatasetsTable();
  }

  function loadDatasetsTable() {
    var tbody = document.querySelector("#datasets-table tbody");
    if (!tbody) return;

    apiFetch("/datasets")
      .then(function (data) {
        var datasets = data.datasets || [];
        if (datasets.length === 0) {
          tbody.innerHTML =
            '<tr><td colspan="4" class="empty-state">No datasets found.</td></tr>';
          return;
        }
        tbody.innerHTML = datasets
          .map(function (ds) {
            return (
              "<tr>" +
              "<td>" +
              escapeHTML(ds.dataset_id || ds.id || "--") +
              "</td>" +
              "<td>" +
              escapeHTML(ds.name || "Unnamed") +
              "</td>" +
              "<td>" +
              createStatusBadge(ds.status || "unknown").outerHTML +
              "</td>" +
              "<td>" +
              '<button class="btn btn-danger btn-sm" onclick="alert(' +
              "'Delete not implemented'" +
              ')">Delete</button>' +
              "</td>" +
              "</tr>"
            );
          })
          .join("");
      })
      .catch(function () {
        tbody.innerHTML =
          '<tr><td colspan="4" class="empty-state">Failed to load datasets.</td></tr>';
      });
  }

  /* ---- Experiments ---- */
  function setupExperiments() {
    var form = getEl("experiment-form");
    if (!form) return;

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var btn = getEl("create-exp-btn");

      var data = {
        name: getEl("exp-name").value.trim(),
        prompt_template: getEl("exp-prompt").value.trim(),
        num_examples: parseInt(getEl("exp-examples").value) || 5,
        temperature: parseFloat(getEl("exp-temp").value) || 0.1,
        top_p: parseFloat(getEl("exp-top-p").value) || 0.95,
        max_tokens: 2048,
        num_branches: parseInt(getEl("exp-branches").value) || 3,
      };

      if (!data.name || !data.prompt_template) return;

      loadingBtn(btn, true);

      apiFetch("/experiments", {
        method: "POST",
        body: JSON.stringify(data),
      })
        .then(function () {
          loadExperimentsTable();
          form.reset();
        })
        .catch(function (err) {
          alert("Error: " + err.message);
        })
        .finally(function () {
          loadingBtn(btn, false);
        });
    });

    loadExperimentsTable();
  }

  function loadExperimentsTable() {
    var tbody = document.querySelector("#experiments-table tbody");
    if (!tbody) return;

    apiFetch("/experiments")
      .then(function (data) {
        var exps = data.experiments || [];
        if (exps.length === 0) {
          tbody.innerHTML =
            '<tr><td colspan="6" class="empty-state">No experiments yet.</td></tr>';
          return;
        }
        tbody.innerHTML = exps
          .map(function (exp) {
            return (
              "<tr>" +
              "<td>" +
              escapeHTML(exp.name || "Unnamed") +
              "</td>" +
              "<td>" +
              createStatusBadge(exp.status || "unknown").outerHTML +
              "</td>" +
              "<td>" +
              formatNumber(exp.metrics?.accuracy) +
              "</td>" +
              "<td>" +
              formatNumber(exp.metrics?.f1) +
              "</td>" +
              "<td>" +
              (exp.metrics?.latency_ms != null
                ? exp.metrics.latency_ms + "ms"
                : "--") +
              "</td>" +
              "<td>" +
              '<button class="btn btn-primary btn-sm" onclick="alert(' +
              "'View details'" +
              ')">View</button>' +
              "</td>" +
              "</tr>"
            );
          })
          .join("");
      })
      .catch(function () {
        tbody.innerHTML =
          '<tr><td colspan="6" class="empty-state">Failed to load.</td></tr>';
      });
  }

  /* ---- Prompts ---- */
  function setupPrompts() {
    var form = getEl("prompt-form");
    if (!form) return;

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var btn = getEl("create-prompt-btn");

      var data = {
        name: getEl("prompt-name").value.trim(),
        content: getEl("prompt-content").value.trim(),
      };

      if (!data.name || !data.content) return;

      loadingBtn(btn, true);

      apiFetch("/prompts", {
        method: "POST",
        body: JSON.stringify(data),
      })
        .then(function () {
          loadPromptsTable();
          form.reset();
        })
        .catch(function (err) {
          alert("Error: " + err.message);
        })
        .finally(function () {
          loadingBtn(btn, false);
        });
    });

    loadPromptsTable();
  }

  function loadPromptsTable() {
    var tbody = document.querySelector("#prompts-table tbody");
    if (!tbody) return;

    apiFetch("/prompts")
      .then(function (data) {
        var prompts = data.prompts || [];
        if (prompts.length === 0) {
          tbody.innerHTML =
            '<tr><td colspan="4" class="empty-state">No prompts yet.</td></tr>';
          return;
        }
        tbody.innerHTML = prompts
          .map(function (p) {
            return (
              "<tr>" +
              "<td>" +
              escapeHTML(p.prompt_id || p.id || "--") +
              "</td>" +
              "<td>" +
              escapeHTML(p.name || "Unnamed") +
              "</td>" +
              "<td>" +
              escapeHTML(
                (p.content || p.template || "").substring(0, 60) + "..."
              ) +
              "</td>" +
              "<td>" +
              '<button class="btn btn-danger btn-sm" onclick="alert(' +
              "'Delete not implemented'" +
              ')">Delete</button>' +
              "</td>" +
              "</tr>"
            );
          })
          .join("");
      })
      .catch(function () {
        tbody.innerHTML =
          '<tr><td colspan="4" class="empty-state">Failed to load.</td></tr>';
      });
  }

  /* ---- Leaderboard ---- */
  function setupLeaderboard() {
    var form = getEl("submit-form");
    if (!form) return;

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      var btn = getEl("submit-btn");
      var status = getEl("submit-status");
      var expId = getEl("submit-exp-id").value.trim();

      if (!expId) return;

      loadingBtn(btn, true);

      fetch(API_BASE + "/leaderboard/submit?experiment_id=" + encodeURIComponent(expId), {
        method: "POST",
      })
        .then(function (r) {
          if (!r.ok) throw new Error("Submission failed");
          return r.json();
        })
        .then(function (res) {
          showAlert(
            status,
            "Submitted: " + res.submission_id + " (" + res.status + ")",
            "success"
          );
          loadLeaderboardTable();
          form.reset();
        })
        .catch(function (err) {
          showAlert(status, err.message, "error");
        })
        .finally(function () {
          loadingBtn(btn, false);
        });
    });

    loadLeaderboardTable();
  }

  function loadLeaderboardTable() {
    var tbody = document.querySelector("#leaderboard-table tbody");
    if (!tbody) return;

    apiFetch("/leaderboard")
      .then(function (data) {
        var rankings = data.rankings || [];
        if (rankings.length === 0) {
          tbody.innerHTML =
            '<tr><td colspan="4" class="empty-state">No submissions yet.</td></tr>';
          return;
        }
        tbody.innerHTML = rankings
          .map(function (entry) {
            return (
              "<tr>" +
              "<td>" +
              entry.rank +
              "</td>" +
              "<td>" +
              escapeHTML(entry.name || entry.experiment_id || "Unnamed") +
              "</td>" +
              "<td>" +
              formatNumber(entry.score) +
              "</td>" +
              "<td>" +
              '<button class="btn btn-primary btn-sm" onclick="alert(' +
              "'View experiment '" +
              ')">View</button>' +
              "</td>" +
              "</tr>"
            );
          })
          .join("");
      })
      .catch(function () {
        tbody.innerHTML =
          '<tr><td colspan="4" class="empty-state">Failed to load.</td></tr>';
      });
  }

  /* ---- Init ---- */
  document.addEventListener("DOMContentLoaded", function () {
    loadDashboard();
    setupAnnotation();
    setupDatasets();
    setupExperiments();
    setupPrompts();
    setupLeaderboard();
  });
})();
