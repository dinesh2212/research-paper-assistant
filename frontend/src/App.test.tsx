import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const healthResponse = {
  status: "ok",
  services: {
    database: { status: "ok", latency_ms: 3.2, detail: "SQLite reachable" },
    qdrant: { status: "ok", latency_ms: 7.1, detail: "Qdrant reachable" },
    ollama: { status: "ok", latency_ms: 12.4, detail: "Ollama reachable" },
  },
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("App", () => {
  const readyPaper = { id: "paper-a", title: "Battery study", status: "ready" };
  const otherPaper = { id: "paper-b", title: "Water study", status: "ready" };
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), { status });

  it("searches only selected ready papers and shows source pages", async () => {
    let resolveSearch: (response: Response) => void = () => {};
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((input) => {
        if (input === "/api/v1/health/ready")
          return Promise.resolve(json(healthResponse));
        if (input === "/api/v1/papers")
          return Promise.resolve(
            json([
              readyPaper,
              otherPaper,
              {
                id: "paper-c",
                title: "Processing study",
                status: "processing",
              },
            ]),
          );
        return new Promise((resolve) => {
          resolveSearch = resolve;
        });
      });
    render(<App />);
    const battery = await screen.findByRole("checkbox", {
      name: "Battery study",
    });
    expect(
      screen.getByRole("checkbox", { name: "Processing study" }),
    ).toBeDisabled();
    const submit = screen.getByRole("button", { name: "Search papers" });
    fireEvent.change(screen.getByLabelText("Your question"), {
      target: { value: "  What changed?  " },
    });
    expect(submit).toBeDisabled();
    fireEvent.click(battery);
    fireEvent.click(submit);
    expect(
      await screen.findByRole("button", { name: "Searching…" }),
    ).toBeDisabled();
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/v1/debug/retrieval",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          question: "What changed?",
          paper_ids: ["paper-a"],
          limit: 8,
        }),
      }),
    );
    resolveSearch(
      json({
        question: "What changed?",
        paper_ids: ["paper-a"],
        candidate_count: 1,
        sources: [
          {
            source_id: 1,
            paper_id: "paper-a",
            paper_title: "Battery study",
            page_number: 3,
            chunk_id: "chunk-a",
            excerpt: "Capacity decreased by ten percent.",
            score: 0.8,
          },
        ],
      }),
    );
    expect(
      await screen.findByText("Capacity decreased by ten percent."),
    ).toBeInTheDocument();
    expect(screen.getByText("Page 3")).toBeInTheDocument();
    expect(screen.getByText("[1] Battery study")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Search papers" })).toBeEnabled();
  });

  it("shows retrieval errors, permits retry, and explains empty results", async () => {
    let attempts = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      if (input === "/api/v1/health/ready")
        return Promise.resolve(json(healthResponse));
      if (input === "/api/v1/papers")
        return Promise.resolve(json([readyPaper, otherPaper]));
      attempts += 1;
      return Promise.resolve(
        attempts === 1
          ? json({ detail: "Retrieval timed out; try again." }, 503)
          : json({
              question: "Evidence?",
              paper_ids: ["paper-a", "paper-b"],
              candidate_count: 0,
              sources: [],
            }),
      );
    });
    render(<App />);
    fireEvent.click(
      await screen.findByRole("checkbox", { name: "Battery study" }),
    );
    fireEvent.click(screen.getByRole("checkbox", { name: "Water study" }));
    fireEvent.change(screen.getByLabelText("Your question"), {
      target: { value: "Evidence?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Search papers" }));
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "Retrieval timed out; try again.",
    );
    fireEvent.click(screen.getByRole("button", { name: "Search papers" }));
    expect(
      await screen.findByText(/No matching passages found/),
    ).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
    expect(globalThis.fetch).toHaveBeenLastCalledWith(
      "/api/v1/debug/retrieval",
      expect.objectContaining({
        body: JSON.stringify({
          question: "Evidence?",
          paper_ids: ["paper-a", "paper-b"],
          limit: 8,
        }),
      }),
    );
  });

  it("uploads a PDF and polls until the paper can be selected", async () => {
    let uploaded = false;
    let polls = 0;
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((input, init) => {
        if (input === "/api/v1/health/ready")
          return Promise.resolve(json(healthResponse));
        if (init?.method === "POST") {
          uploaded = true;
          return Promise.resolve(
            json(
              { ...readyPaper, status: "pending", error_message: null },
              201,
            ),
          );
        }
        if (!uploaded) return Promise.resolve(json([]));
        polls += 1;
        return Promise.resolve(
          json([
            { ...readyPaper, status: polls === 1 ? "processing" : "ready" },
          ]),
        );
      });
    render(<App />);
    await screen.findByText("All systems ready");
    const file = new File(["%PDF-1.7"], "study.pdf", {
      type: "application/pdf",
    });
    fireEvent.change(screen.getByLabelText("Upload a PDF"), {
      target: { files: [file] },
    });
    expect(
      await screen.findByRole("checkbox", { name: "Battery study" }),
    ).toBeDisabled();
    const uploadCall = fetchMock.mock.calls.find(
      ([, init]) => init?.method === "POST",
    );
    expect((uploadCall?.[1]?.body as FormData).get("file")).toBe(file);
    await waitFor(
      () =>
        expect(
          screen.getByRole("checkbox", { name: "Battery study" }),
        ).toBeEnabled(),
      { timeout: 4000 },
    );
  });

  it("shows upload errors and ingestion failure details", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      if (input === "/api/v1/health/ready")
        return Promise.resolve(json(healthResponse));
      if (init?.method === "POST")
        return Promise.resolve(
          json({ detail: "This PDF has already been uploaded." }, 409),
        );
      if (input === "/api/v1/papers/paper-a")
        return Promise.resolve(
          json({
            ...readyPaper,
            status: "failed",
            error_message: "No extractable text. OCR is required.",
          }),
        );
      return Promise.resolve(json([{ ...readyPaper, status: "failed" }]));
    });
    render(<App />);
    expect(
      await screen.findByRole("checkbox", { name: "Battery study" }),
    ).toBeDisabled();
    fireEvent.click(
      screen.getByRole("button", { name: "Why did processing fail?" }),
    );
    expect(
      await screen.findByText("No extractable text. OCR is required."),
    ).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Upload a PDF"), {
      target: { files: [new File(["%PDF"], "study.pdf")] },
    });
    expect(await screen.findByRole("alert")).toHaveTextContent(
      "This PDF has already been uploaded.",
    );
    expect(screen.getByLabelText("Upload a PDF")).toBeEnabled();
  });

  it("shows an empty library and healthy local services", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url =
        typeof input === "string"
          ? input
          : input instanceof URL
            ? input.href
            : input.url;
      const payload = url.endsWith("/health/ready") ? healthResponse : [];
      return Promise.resolve(
        new Response(JSON.stringify(payload), { status: 200 }),
      );
    });

    render(<App />);

    expect(screen.getByText("Loading your workspace…")).toBeInTheDocument();
    expect(await screen.findByText("All systems ready")).toBeInTheDocument();
    expect(screen.getByText("Your library is ready.")).toBeInTheDocument();
    expect(screen.getByText("SQLite")).toBeInTheDocument();
    expect(screen.getByText("Qdrant")).toBeInTheDocument();
    expect(screen.getByText("Ollama")).toBeInTheDocument();
  });

  it("offers a retry when the backend cannot be reached", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new Error("Connection refused"),
    );

    render(<App />);

    expect(await screen.findByRole("alert")).toHaveTextContent(
      "The backend is offline.",
    );
    expect(
      screen.getByRole("button", { name: "Try again" }),
    ).toBeInTheDocument();

    await waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(2));
  });
});
