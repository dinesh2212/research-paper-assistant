import { useEffect, useState } from "react";

import {
  getPaper,
  getPapers,
  searchPapers,
  uploadPaper,
  type PaperSummary,
  type RetrievalResponse,
} from "./api";

interface Props {
  papers: PaperSummary[];
  onPapersChange: (papers: PaperSummary[]) => void;
  available: boolean;
}

function message(error: unknown): string {
  return error instanceof Error
    ? error.message
    : "The request failed. Please try again.";
}

export default function PaperWorkspace({
  papers,
  onPapersChange,
  available,
}: Props) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [uploading, setUploading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [libraryError, setLibraryError] = useState("");
  const [searchError, setSearchError] = useState("");
  const [notice, setNotice] = useState("");
  const [result, setResult] = useState<RetrievalResponse | null>(null);
  const [failures, setFailures] = useState<Record<string, string>>({});

  // Poll only while ingestion is active; cleanup prevents stale responses
  // from replacing a newer library after uploads or a workspace refresh.
  const pending = papers.some((paper) =>
    ["pending", "processing"].includes(paper.status),
  );
  useEffect(() => {
    if (!pending || !available) return;
    let current = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const updated = await getPapers();
        if (current) {
          setLibraryError("");
          onPapersChange(updated);
        }
      } catch (error) {
        if (current) {
          setLibraryError(
            `Could not refresh processing status: ${message(error)}`,
          );
          timer = setTimeout(() => void poll(), 3000);
        }
      }
    };
    timer = setTimeout(() => void poll(), 2000);
    return () => {
      current = false;
      clearTimeout(timer);
    };
  }, [papers, pending, available, onPapersChange]);

  const selected = selectedIds.filter((id) =>
    papers.some((paper) => paper.id === id && paper.status === "ready"),
  );

  async function upload(file: File) {
    setUploading(true);
    setLibraryError("");
    setNotice("");
    try {
      const paper = await uploadPaper(file);
      const updated = await getPapers();
      onPapersChange(updated);
      setNotice(`${paper.title} uploaded. Select it once its status is ready.`);
    } catch (error) {
      setLibraryError(message(error));
    } finally {
      setUploading(false);
    }
  }

  async function showFailure(id: string) {
    try {
      const paper = await getPaper(id);
      setFailures((previous) => ({
        ...previous,
        [id]: paper.error_message ?? "Processing failed.",
      }));
    } catch (error) {
      setLibraryError(message(error));
    }
  }

  async function search() {
    if (searching || !available || !question.trim() || selected.length === 0)
      return;
    setSearching(true);
    setSearchError("");
    setResult(null);
    try {
      setResult(await searchPapers(question.trim(), selected));
    } catch (error) {
      setSearchError(message(error));
    } finally {
      setSearching(false);
    }
  }

  return (
    <div className="research-panels">
      <article className="panel library-panel" id="library">
        <div className="panel-heading">
          <div>
            <span className="section-number">01</span>
            <h2>Paper library</h2>
          </div>
          <span className="count-badge">{papers.length}</span>
        </div>
        <div className="upload-control">
          <label htmlFor="paper-upload">Upload a PDF</label>
          <input
            id="paper-upload"
            type="file"
            accept=".pdf,application/pdf"
            disabled={!available || uploading}
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              event.currentTarget.value = "";
              if (file) void upload(file);
            }}
          />
          <p>Choose a text-based research paper with selectable text.</p>
        </div>
        {(uploading || notice) && (
          <p role="status">{uploading ? "Uploading PDF…" : notice}</p>
        )}
        {libraryError && (
          <p className="inline-error" role="alert">
            {libraryError}
          </p>
        )}
        {available && papers.length === 0 && (
          <div className="empty-state">
            <p>Your library is ready.</p>
            <span>Upload a PDF to start searching its pages.</span>
          </div>
        )}
        {papers.length > 0 && (
          <>
            <p className="selection-hint">
              Select up to 20 ready papers to search. {selected.length}{" "}
              selected.
            </p>
            <ul className="paper-list">
              {papers.map((paper) => (
                <li key={paper.id} className="paper-row">
                  <label>
                    <input
                      type="checkbox"
                      checked={selected.includes(paper.id)}
                      disabled={
                        !available ||
                        searching ||
                        paper.status !== "ready" ||
                        (selected.length >= 20 && !selected.includes(paper.id))
                      }
                      onChange={(event) =>
                        setSelectedIds(
                          event.target.checked
                            ? [...selected, paper.id]
                            : selected.filter((id) => id !== paper.id),
                        )
                      }
                    />
                    <span>{paper.title}</span>
                  </label>
                  <small>{paper.status}</small>
                  {paper.status === "failed" && (
                    <div className="paper-failure">
                      <button
                        type="button"
                        className="refresh-button"
                        onClick={() => void showFailure(paper.id)}
                      >
                        Why did processing fail?
                      </button>
                      {failures[paper.id] && <p>{failures[paper.id]}</p>}
                    </div>
                  )}
                </li>
              ))}
            </ul>
          </>
        )}
      </article>

      <article className="panel query-panel" id="query">
        <div className="panel-heading">
          <div>
            <span className="section-number">02</span>
            <h2>Search your papers</h2>
          </div>
        </div>
        <p className="query-help" id="query-help">
          Ask a question to find supporting passages and page references.
        </p>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            void search();
          }}
        >
          <label htmlFor="question">Your question</label>
          <textarea
            id="question"
            value={question}
            maxLength={2000}
            rows={4}
            aria-describedby="query-help selection-help"
            placeholder="What are the main findings?"
            disabled={searching}
            required
            onChange={(event) => setQuestion(event.target.value)}
          />
          <div className="query-actions">
            <button
              type="submit"
              className="primary-button"
              disabled={
                !available ||
                searching ||
                !question.trim() ||
                selected.length === 0
              }
            >
              {searching ? "Searching…" : "Search papers"}
            </button>
            <span id="selection-help">
              {selected.length === 0
                ? "Select a ready paper above to search."
                : `Searching ${selected.length} selected ${selected.length === 1 ? "paper" : "papers"}.`}
            </span>
          </div>
        </form>
        {searching && <p role="status">Finding relevant passages…</p>}
        {searchError && (
          <p role="alert" className="inline-error">
            {searchError}
          </p>
        )}
        {result && (
          <section className="search-results" aria-label="Search results">
            <h3>Results for “{result.question}”</h3>
            <p role="status">
              {result.sources.length === 0
                ? "No matching passages found. Try another question or select different papers."
                : `${result.sources.length} supporting ${result.sources.length === 1 ? "passage" : "passages"} found.`}
            </p>
            <ol className="source-list">
              {result.sources.map((source) => (
                <li key={source.source_id}>
                  <div className="source-heading">
                    <strong>
                      [{source.source_id}] {source.paper_title}
                    </strong>
                    <span>Page {source.page_number}</span>
                  </div>
                  <blockquote>{source.excerpt}</blockquote>
                </li>
              ))}
            </ol>
          </section>
        )}
      </article>
    </div>
  );
}
