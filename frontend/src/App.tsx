import { useCallback, useEffect, useState } from "react";

import {
  getPapers,
  getReadiness,
  type PaperSummary,
  type ReadinessResponse,
} from "./api";

import PaperWorkspace from "./PaperWorkspace";

type LoadState = "loading" | "ready" | "error";

const serviceLabels: Record<string, string> = {
  database: "SQLite",
  qdrant: "Qdrant",
  ollama: "Ollama",
};

function App() {
  const [readiness, setReadiness] = useState<ReadinessResponse | null>(null);
  const [papers, setPapers] = useState<PaperSummary[]>([]);
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [errorMessage, setErrorMessage] = useState("");

  const loadWorkspace = useCallback(async () => {
    setLoadState("loading");
    setErrorMessage("");

    try {
      const [healthResult, papersResult] = await Promise.all([
        getReadiness(),
        getPapers(),
      ]);
      setReadiness(healthResult);
      setPapers(papersResult);
      setLoadState("ready");
    } catch (error) {
      setReadiness(null);
      setLoadState("error");
      setErrorMessage(
        error instanceof Error
          ? error.message
          : "The backend could not be reached.",
      );
    }
  }, []);

  useEffect(() => {
    let isCurrent = true;

    void Promise.all([getReadiness(), getPapers()])
      .then(([healthResult, papersResult]) => {
        if (!isCurrent) return;
        setReadiness(healthResult);
        setPapers(papersResult);
        setLoadState("ready");
      })
      .catch((error: unknown) => {
        if (!isCurrent) return;
        setReadiness(null);
        setLoadState("error");
        setErrorMessage(
          error instanceof Error
            ? error.message
            : "The backend could not be reached.",
        );
      });

    return () => {
      isCurrent = false;
    };
  }, []);

  const services = readiness ? Object.entries(readiness.services) : [];
  const systemReady = loadState === "ready" && readiness?.status === "ok";

  return (
    <main className="app-shell">
      <header className="topbar">
        <a
          className="brand"
          href="#top"
          aria-label="Research Paper Assistant home"
        >
          <span className="brand-mark" aria-hidden="true">
            R
          </span>
          <span>
            Research Paper
            <strong>Assistant</strong>
          </span>
        </a>
        <span className="local-badge">
          <span className="local-dot" /> Local workspace
        </span>
      </header>

      <section className="hero" id="top">
        <div className="eyebrow">Citation-grounded research</div>
        <h1>
          Search your papers.
          <br />
          Find the evidence.
        </h1>
        <p>
          Upload research papers, ask a question, and explore relevant passages
          with references to the original pages.
        </p>
        <div className="hero-actions">
          <a className="primary-button" href="#library">
            Upload a paper
          </a>
          <a href="#query">Search your papers</a>
        </div>
      </section>

      <section className="workspace-grid" aria-label="Workspace overview">
        <div className="workspace-main">
          {loadState === "loading" && (
            <p className="panel-message">Loading your workspace…</p>
          )}
          {loadState === "error" && (
            <div className="error-state" role="alert">
              <p>The backend is offline.</p>
              <span>{errorMessage}</span>
              <button type="button" onClick={() => void loadWorkspace()}>
                Try again
              </button>
            </div>
          )}
          <PaperWorkspace
            papers={papers}
            onPapersChange={setPapers}
            available={loadState === "ready"}
          />
        </div>

        <article className="panel status-panel">
          <div className="panel-heading">
            <div>
              <span className="section-number">03</span>
              <h2>System status</h2>
            </div>
            <button
              className="refresh-button"
              type="button"
              onClick={() => void loadWorkspace()}
              disabled={loadState === "loading"}
            >
              Refresh
            </button>
          </div>

          <div
            className={`overall-status ${systemReady ? "is-ready" : "is-waiting"}`}
          >
            <span className="status-light" />
            <div>
              <strong>
                {systemReady ? "All systems ready" : "Waiting for services"}
              </strong>
              <span>
                {systemReady
                  ? "Your local AI stack is available."
                  : "Checking the local application stack."}
              </span>
            </div>
          </div>

          <ul className="service-list">
            {services.map(([name, service]) => (
              <li key={name}>
                <div>
                  <span className={`service-dot ${service.status}`} />
                  <span>{serviceLabels[name] ?? name}</span>
                </div>
                <small>
                  {service.status === "ok"
                    ? `${String(service.latency_ms)} ms`
                    : "Offline"}
                </small>
              </li>
            ))}
            {services.length === 0 &&
              ["SQLite", "Qdrant", "Ollama"].map((name) => (
                <li key={name} className="service-placeholder">
                  <div>
                    <span className="service-dot" />
                    <span>{name}</span>
                  </div>
                  <small>—</small>
                </li>
              ))}
          </ul>
        </article>
      </section>

      <footer>
        <span>Private by design</span>
        <span>Files and models stay on this machine</span>
      </footer>
    </main>
  );
}

export default App;
