import { useRouter } from "next/router";
import { useEffect, useMemo, useState } from "react";
import {
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import Layout from "../components/Layout";
import type { ActiveCheckpoint, ExperimentRow } from "../lib/types";

type SortKey =
  | "run_id"
  | "timestamp"
  | "decision"
  | "discard_reason"
  | "research_val_scored_map50_95"
  | "locked_eval_scored_map50_95"
  | "locked_eval_nodule_cyst_recall";

type SortState = {
  key: SortKey;
  direction: "asc" | "desc";
};

type TooltipPayload = {
  payload?: ExperimentRow;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const POLL_MS = 30_000;

const decisionColors: Record<ExperimentRow["decision"], string> = {
  KEEP: "#16a34a",
  DISCARD: "#78716c",
  FAILED: "#dc2626",
  PENDING_KEEP_RULE: "#2563eb",
};

function formatNumber(value: number | null | undefined, digits = 4) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

function formatTimestamp(value: string) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

async function fetchJson<T>(path: string) {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed: ${path}`);
  }
  return (await response.json()) as T;
}

function latestKept(rows: ExperimentRow[]) {
  const kept = rows.filter((row) => row.decision === "KEEP");
  return kept.sort((a, b) => {
    const aTime = new Date(a.timestamp).getTime();
    const bTime = new Date(b.timestamp).getTime();
    return (Number.isNaN(bTime) ? 0 : bTime) - (Number.isNaN(aTime) ? 0 : aTime);
  })[0];
}

function valueForSort(row: ExperimentRow, key: SortKey) {
  return row[key] ?? "";
}

function sortedRows(rows: ExperimentRow[], sort: SortState) {
  return [...rows].sort((a, b) => {
    const aValue = valueForSort(a, sort.key);
    const bValue = valueForSort(b, sort.key);
    const direction = sort.direction === "asc" ? 1 : -1;

    if (typeof aValue === "number" && typeof bValue === "number") {
      return (aValue - bValue) * direction;
    }
    return String(aValue).localeCompare(String(bValue)) * direction;
  });
}

function chartDomain(rows: ExperimentRow[]) {
  const values = rows.flatMap((row) => [
    row.research_val_scored_map50_95,
    row.locked_eval_scored_map50_95,
  ]);
  const maxValue = Math.max(0.1, ...values.filter((value) => Number.isFinite(value)));
  return [0, Math.ceil(maxValue * 100) / 100] as [number, number];
}

function ActiveCheckpointCard({ checkpoint }: { checkpoint: ActiveCheckpoint | null }) {
  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-stone-950">Active checkpoint</h2>
          <div className="mt-4 grid gap-3 text-sm text-stone-700 md:grid-cols-2">
            <p>
              <span className="font-semibold text-stone-950">weights_path:</span>{" "}
              {checkpoint?.weights_path ?? "-"}
            </p>
            <p>
              <span className="font-semibold text-stone-950">iteration_id:</span>{" "}
              {checkpoint?.iteration_id ?? "-"}
            </p>
            <p>
              <span className="font-semibold text-stone-950">preprocessing:</span>{" "}
              {checkpoint?.preprocessing ?? "-"}
            </p>
            <p>
              <span className="font-semibold text-stone-950">weights_exists:</span>{" "}
              {checkpoint ? String(checkpoint.weights_exists) : "-"}
            </p>
          </div>
        </div>
        {checkpoint && !checkpoint.weights_exists ? (
          <p className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm font-semibold text-stone-900">
            No active checkpoint loaded — Doctor/Patient tabs are inactive.
          </p>
        ) : null}
      </div>
    </section>
  );
}

function LatestKeptCard({
  checkpoint,
  kept,
}: {
  checkpoint: ActiveCheckpoint | null;
  kept: ExperimentRow | undefined;
}) {
  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <h2 className="text-xl font-semibold text-stone-950">Latest kept iteration</h2>
      {kept ? (
        <div className="mt-3 space-y-3 text-sm leading-6 text-stone-700">
          <p>
            Latest loop winner: <span className="font-semibold text-stone-950">{kept.run_id}</span>{" "}
            (locked_eval mAP {formatNumber(kept.locked_eval_scored_map50_95)}).
          </p>
          {kept.run_id !== checkpoint?.iteration_id ? (
            <p className="rounded-md border border-sky-200 bg-sky-50 px-3 py-2 font-medium text-sky-950">
              Latest loop winner: {kept.run_id} (locked_eval mAP{" "}
              {formatNumber(kept.locked_eval_scored_map50_95)}). Active demo checkpoint:{" "}
              {checkpoint?.iteration_id ?? "none"}. To promote, edit config/active_checkpoint.json.
            </p>
          ) : (
            <p className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 font-medium text-emerald-950">
              Active demo checkpoint matches the latest kept iteration.
            </p>
          )}
        </div>
      ) : (
        <p className="mt-3 rounded-md border border-stone-200 bg-stone-50 px-3 py-2 text-sm text-stone-700">
          No kept iteration has arrived yet.
        </p>
      )}
    </section>
  );
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  if (!active || !payload?.length || !payload[0].payload) {
    return null;
  }
  const row = payload[0].payload;
  return (
    <div className="rounded-md border border-stone-200 bg-white p-3 text-sm shadow-sm">
      <p className="font-semibold text-stone-950">{row.run_id}</p>
      <p className="text-stone-700">{row.decision}</p>
      <p className="text-stone-700">research_val: {formatNumber(row.research_val_scored_map50_95)}</p>
      <p className="text-stone-700">locked_eval: {formatNumber(row.locked_eval_scored_map50_95)}</p>
    </div>
  );
}

function ResearchScatter({ rows }: { rows: ExperimentRow[] }) {
  const router = useRouter();
  const domain = chartDomain(rows);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <h2 className="text-xl font-semibold text-stone-950">Validation trajectory</h2>
        <div className="flex flex-wrap gap-3 text-sm text-stone-700">
          {Object.entries(decisionColors).map(([decision, color]) => (
            <span key={decision} className="inline-flex items-center gap-2">
              <span className="h-3 w-3 rounded-full" style={{ background: color }} />
              {decision}
            </span>
          ))}
        </div>
      </div>
      <div className="mt-4 h-[420px] w-full min-w-0">
        {mounted ? (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 16, right: 24, bottom: 24, left: 10 }}>
              <CartesianGrid stroke="#e7e5e4" />
              <XAxis
                type="number"
                dataKey="research_val_scored_map50_95"
                name="research_val"
                domain={domain}
                tickFormatter={(value) => formatNumber(Number(value), 2)}
                label={{ value: "research_val mAP50-95", position: "bottom", offset: 8 }}
              />
              <YAxis
                type="number"
                dataKey="locked_eval_scored_map50_95"
                name="locked_eval"
                domain={domain}
                tickFormatter={(value) => formatNumber(Number(value), 2)}
                label={{ value: "locked_eval mAP50-95", angle: -90, position: "insideLeft" }}
              />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<CustomTooltip />} />
              <ReferenceLine
                segment={[
                  { x: domain[0], y: domain[0] },
                  { x: domain[1], y: domain[1] },
                ]}
                stroke="#0f766e"
                strokeDasharray="4 4"
              />
              <Scatter
                data={rows}
                onClick={(point) => {
                  const row = ((point as { payload?: ExperimentRow }).payload ?? point) as ExperimentRow;
                  if (row.run_id) {
                    void router.push(`/research/iteration/${row.run_id}`);
                  }
                }}
              >
                {rows.map((row) => (
                  <Cell key={row.run_id} fill={decisionColors[row.decision] ?? "#2563eb"} cursor="pointer" />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <div className="flex h-full items-center justify-center rounded-md bg-stone-50 text-sm text-stone-600">
            Loading chart...
          </div>
        )}
      </div>
    </section>
  );
}

function IterationsTable({ rows }: { rows: ExperimentRow[] }) {
  const router = useRouter();
  const [sort, setSort] = useState<SortState>({
    key: "timestamp",
    direction: "desc",
  });
  const orderedRows = useMemo(() => sortedRows(rows, sort), [rows, sort]);

  function toggleSort(key: SortKey) {
    setSort((current) => ({
      key,
      direction: current.key === key && current.direction === "desc" ? "asc" : "desc",
    }));
  }

  const columns: Array<{ key: SortKey; label: string; numeric?: boolean }> = [
    { key: "run_id", label: "run_id" },
    { key: "timestamp", label: "timestamp" },
    { key: "decision", label: "decision" },
    { key: "discard_reason", label: "discard_reason" },
    { key: "research_val_scored_map50_95", label: "research_val_mAP", numeric: true },
    { key: "locked_eval_scored_map50_95", label: "locked_eval_mAP", numeric: true },
    { key: "locked_eval_nodule_cyst_recall", label: "nodule_recall", numeric: true },
  ];

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <h2 className="text-xl font-semibold text-stone-950">Iterations</h2>
      <div className="mt-4 overflow-x-auto rounded-md border border-stone-200">
        <table className="w-full min-w-[900px] border-collapse text-left text-sm">
          <thead className="bg-stone-100 text-stone-700">
            <tr>
              {columns.map((column) => (
                <th key={column.key} className={column.numeric ? "px-3 py-2 text-right" : "px-3 py-2"}>
                  <button
                    type="button"
                    onClick={() => toggleSort(column.key)}
                    className="font-semibold text-stone-800 hover:text-emerald-800"
                  >
                    {column.label}
                    {sort.key === column.key ? ` ${sort.direction}` : ""}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {orderedRows.map((row) => (
              <tr
                key={row.run_id}
                onClick={() => void router.push(`/research/iteration/${row.run_id}`)}
                className="cursor-pointer border-t border-stone-200 hover:bg-rose-50"
              >
                <td className="px-3 py-2 font-semibold text-stone-950">{row.run_id}</td>
                <td className="px-3 py-2 text-stone-700">{formatTimestamp(row.timestamp)}</td>
                <td className="px-3 py-2 text-stone-700">{row.decision}</td>
                <td className="max-w-sm px-3 py-2 text-stone-700">{row.discard_reason ?? "-"}</td>
                <td className="px-3 py-2 text-right text-stone-700">
                  {formatNumber(row.research_val_scored_map50_95)}
                </td>
                <td className="px-3 py-2 text-right text-stone-700">
                  {formatNumber(row.locked_eval_scored_map50_95)}
                </td>
                <td className="px-3 py-2 text-right text-stone-700">
                  {formatNumber(row.locked_eval_nodule_cyst_recall)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function ResearchPage() {
  const [checkpoint, setCheckpoint] = useState<ActiveCheckpoint | null>(null);
  const [rows, setRows] = useState<ExperimentRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [nextCheckpoint, nextRows] = await Promise.all([
          fetchJson<ActiveCheckpoint>("/api/active_checkpoint"),
          fetchJson<ExperimentRow[]>("/api/iterations"),
        ]);
        if (!cancelled) {
          setCheckpoint(nextCheckpoint);
          setRows(nextRows);
          setError(null);
          setLoading(false);
        }
      } catch (error) {
        if (!cancelled) {
          setError(error instanceof Error ? error.message : "Unable to load research status.");
          setLoading(false);
        }
      }
    }

    void load();
    const timer = window.setInterval(() => void load(), POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const kept = useMemo(() => latestKept(rows), [rows]);

  return (
    <Layout activeTab="research">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-semibold tracking-normal text-stone-950">
            Research — Autoresearch Loop Status
          </h1>
        </div>
        {loading ? (
          <div className="rounded-md border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-medium text-sky-950">
            Loading loop status...
          </div>
        ) : null}
        {error ? (
          <div className="rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm font-medium text-stone-900">
            {error}
          </div>
        ) : null}
        <ActiveCheckpointCard checkpoint={checkpoint} />
        <LatestKeptCard checkpoint={checkpoint} kept={kept} />
        <ResearchScatter rows={rows} />
        <IterationsTable rows={rows} />
      </div>
    </Layout>
  );
}
