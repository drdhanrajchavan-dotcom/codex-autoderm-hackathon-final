import dynamic from "next/dynamic";
import { useRouter } from "next/router";
import { useEffect, useMemo, useState } from "react";

import Layout from "../../../components/Layout";
import type { ExperimentRow, IterationDetail } from "../../../lib/types";

const ReactDiffViewer = dynamic(() => import("react-diff-viewer"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

function formatNumber(value: unknown, digits = 4) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(digits) : "-";
}

function isExperimentRow(value: IterationDetail["experiment_row"]): value is ExperimentRow {
  return Boolean(value && typeof value === "object" && "run_id" in value);
}

function formatMetricValue(value: unknown) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (typeof value === "boolean") {
    return String(value);
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

async function fetchIteration(runId: string) {
  const response = await fetch(`${API_BASE}/api/iteration/${encodeURIComponent(runId)}`, {
    cache: "no-store",
  });
  if (!response.ok) {
    throw new Error(`Unable to load iteration: ${runId}`);
  }
  return (await response.json()) as IterationDetail;
}

function splitUnifiedDiff(diffText: string) {
  const oldLines: string[] = [];
  const newLines: string[] = [];

  for (const line of diffText.split("\n")) {
    if (line.startsWith("--- ") || line.startsWith("+++ ") || line.startsWith("@@")) {
      continue;
    }
    if (line.startsWith("-")) {
      oldLines.push(line.slice(1));
    } else if (line.startsWith("+")) {
      newLines.push(line.slice(1));
    } else if (line.startsWith(" ")) {
      oldLines.push(line.slice(1));
      newLines.push(line.slice(1));
    }
  }

  return {
    oldValue: oldLines.join("\n"),
    newValue: newLines.join("\n"),
  };
}

function MetricsCard({ row }: { row: IterationDetail["experiment_row"] }) {
  if (!isExperimentRow(row)) {
    return (
      <section className="rounded-md border border-stone-200 bg-white p-4">
        <h2 className="text-xl font-semibold text-stone-950">Metrics</h2>
        <p className="mt-3 text-sm text-stone-700">No experiment row was found for this iteration.</p>
      </section>
    );
  }

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <h2 className="text-xl font-semibold text-stone-950">Metrics</h2>
      <div className="mt-4 grid gap-3 md:grid-cols-2 lg:grid-cols-3">
        <div className="rounded-md bg-stone-50 px-3 py-2">
          <p className="text-sm text-stone-600">run_id</p>
          <p className="font-semibold text-stone-950">{row.run_id}</p>
        </div>
        <div className="rounded-md bg-stone-50 px-3 py-2">
          <p className="text-sm text-stone-600">decision</p>
          <p className="font-semibold text-stone-950">{row.decision}</p>
        </div>
        <div className="rounded-md bg-stone-50 px-3 py-2">
          <p className="text-sm text-stone-600">research_val mAP</p>
          <p className="font-semibold text-stone-950">{formatNumber(row.research_val_scored_map50_95)}</p>
        </div>
        <div className="rounded-md bg-stone-50 px-3 py-2">
          <p className="text-sm text-stone-600">locked_eval mAP</p>
          <p className="font-semibold text-stone-950">{formatNumber(row.locked_eval_scored_map50_95)}</p>
        </div>
        <div className="rounded-md bg-stone-50 px-3 py-2">
          <p className="text-sm text-stone-600">nodule_cyst recall</p>
          <p className="font-semibold text-stone-950">{formatNumber(row.locked_eval_nodule_cyst_recall)}</p>
        </div>
        <div className="rounded-md bg-stone-50 px-3 py-2">
          <p className="text-sm text-stone-600">discard_reason</p>
          <p className="font-semibold text-stone-950">{row.discard_reason ?? "-"}</p>
        </div>
      </div>
      <div className="mt-4 overflow-x-auto rounded-md border border-stone-200">
        <table className="w-full min-w-[720px] border-collapse text-left text-sm">
          <tbody>
            {Object.entries(row).map(([key, value]) => (
              <tr key={key} className="border-t border-stone-200 first:border-t-0">
                <th className="w-64 bg-stone-50 px-3 py-2 font-semibold text-stone-800">{key}</th>
                <td className="px-3 py-2 text-stone-700">{formatMetricValue(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CollapsibleText({
  title,
  text,
  open = false,
}: {
  title: string;
  text: string;
  open?: boolean;
}) {
  return (
    <details open={open} className="rounded-md border border-stone-200 bg-white p-4">
      <summary className="cursor-pointer text-xl font-semibold text-stone-950">{title}</summary>
      <pre className="mt-4 max-h-[520px] overflow-auto rounded-md bg-stone-950 p-4 text-sm leading-6 text-stone-50">
        {text || "No content found."}
      </pre>
    </details>
  );
}

function DiffSection({ diffText }: { diffText: string }) {
  const values = useMemo(() => splitUnifiedDiff(diffText), [diffText]);

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <h2 className="text-xl font-semibold text-stone-950">train.py diff</h2>
      <div className="mt-4 overflow-x-auto rounded-md border border-stone-200 text-sm">
        {diffText ? (
          <ReactDiffViewer
            oldValue={values.oldValue}
            newValue={values.newValue}
            splitView
            useDarkTheme={false}
            hideLineNumbers={false}
          />
        ) : (
          <p className="p-4 text-sm text-stone-700">No train.py diff found.</p>
        )}
      </div>
    </section>
  );
}

export default function IterationPage() {
  const router = useRouter();
  const runId = typeof router.query.run_id === "string" ? router.query.run_id : "";
  const [detail, setDetail] = useState<IterationDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!runId) {
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    void fetchIteration(runId)
      .then((nextDetail) => {
        if (!cancelled) {
          setDetail(nextDetail);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setError(error instanceof Error ? error.message : "Unable to load iteration.");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [runId]);

  return (
    <Layout activeTab="research" pageTitle="AutoDerm — Iteration">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-semibold tracking-normal text-stone-950">
            Research Iteration
          </h1>
          {runId ? <p className="mt-2 text-base text-stone-700">{runId}</p> : null}
        </div>
        {loading ? (
          <div className="rounded-md border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-medium text-sky-950">
            Loading iteration details...
          </div>
        ) : null}
        {error ? (
          <div className="rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm font-medium text-stone-900">
            {error}
          </div>
        ) : null}
        {detail ? (
          <>
            <MetricsCard row={detail.experiment_row} />
            <CollapsibleText title="Codex prompt" text={detail.prompt_text} />
            <DiffSection diffText={detail.train_diff_text} />
            <CollapsibleText
              title="experiment_row.json"
              text={JSON.stringify(detail.experiment_row, null, 2)}
            />
            <CollapsibleText title="worker_stdout.log tail" text={detail.worker_stdout_tail} open />
            <CollapsibleText title="Codex response" text={detail.codex_response_text} />
          </>
        ) : null}
      </div>
    </Layout>
  );
}
