import type { GetStaticProps } from "next";
import { useRouter } from "next/router";
import { useEffect, useMemo, useRef, useState } from "react";
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
import OverlayCanvas from "../components/OverlayCanvas";
import type {
  ActiveCheckpoint,
  AnyClass,
  ExperimentRow,
  HayashiBadge,
  InferenceCheckpointOption,
  InferenceResponse,
  IterationCompareResponse,
  ScoredClass,
} from "../lib/types";
import { SCORED_CLASSES } from "../lib/types";

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

type DemoPhoto = {
  label: string;
  url: string;
};

type ImageSelection = {
  file: File;
  url: string;
};

type ResearchPageProps = {
  demoPhotos: DemoPhoto[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
const POLL_MS = 30_000;

const decisionColors: Record<ExperimentRow["decision"], string> = {
  KEEP: "#16a34a",
  DISCARD: "#78716c",
  FAILED: "#dc2626",
  PENDING_KEEP_RULE: "#2563eb",
};

const classLabels: Record<AnyClass, string> = {
  comedone_open: "Open comedones",
  comedone_closed: "Closed comedones",
  papule: "Papules",
  pustule: "Pustules",
  nodule_cyst: "Nodules or cysts",
  post_acne_mark: "Post-acne marks",
};

const badgeClasses: Record<HayashiBadge, string> = {
  clear: "border-emerald-300 bg-emerald-50 text-emerald-900",
  almost_clear: "border-lime-300 bg-lime-50 text-lime-900",
  mild: "border-yellow-300 bg-yellow-50 text-yellow-900",
  moderate: "border-rose-300 bg-rose-50 text-rose-900",
  severe: "border-red-300 bg-red-50 text-red-900",
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

function makeObjectUrl(file: File) {
  return URL.createObjectURL(file);
}

function revokeSelection(selection: ImageSelection | null) {
  if (selection) {
    URL.revokeObjectURL(selection.url);
  }
}

async function fileFromDemoPhoto(photo: DemoPhoto) {
  const response = await fetch(photo.url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Unable to load sample photo.");
  }
  const blob = await response.blob();
  return new File([blob], photo.label, { type: blob.type || "image/jpeg" });
}

async function compareIterations(file: File, iterationA: string, iterationB: string) {
  const body = new FormData();
  body.append("image", file);
  body.append("iteration_a", iterationA);
  body.append("iteration_b", iterationB);
  const response = await fetch(`${API_BASE}/api/compare_iterations`, {
    method: "POST",
    body,
    cache: "no-store",
  });
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? "Iteration comparison failed.");
  }
  return (await response.json()) as IterationCompareResponse;
}

function latestKept(rows: ExperimentRow[]) {
  const kept = rows.filter((row) => row.decision === "KEEP");
  return kept.sort((a, b) => {
    const aTime = new Date(a.timestamp).getTime();
    const bTime = new Date(b.timestamp).getTime();
    return (Number.isNaN(bTime) ? 0 : bTime) - (Number.isNaN(aTime) ? 0 : aTime);
  })[0];
}

function usableOptions(options: InferenceCheckpointOption[]) {
  return options.filter((option) => option.usable);
}

function latestKeptOption(options: InferenceCheckpointOption[]) {
  return usableOptions(options)
    .filter((option) => option.decision === "KEEP")
    .sort((a, b) => {
      const aTime = new Date(a.timestamp).getTime();
      const bTime = new Date(b.timestamp).getTime();
      return (Number.isNaN(bTime) ? 0 : bTime) - (Number.isNaN(aTime) ? 0 : aTime);
    })[0];
}

function highestLockedEvalOption(options: InferenceCheckpointOption[], excludedRunId?: string) {
  return usableOptions(options)
    .filter((option) => option.run_id !== excludedRunId)
    .sort((a, b) => b.locked_eval_scored_map50_95 - a.locked_eval_scored_map50_95)[0];
}

function defaultIterationA(
  options: InferenceCheckpointOption[],
  checkpoint: ActiveCheckpoint | null,
) {
  const activeOption = usableOptions(options).find((option) => option.run_id === checkpoint?.iteration_id);
  return activeOption ?? latestKeptOption(options) ?? usableOptions(options)[0];
}

function defaultIterationB(options: InferenceCheckpointOption[], iterationA?: string) {
  return highestLockedEvalOption(options, iterationA) ?? usableOptions(options).find((option) => option.run_id !== iterationA);
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
  const isPoweringDemo = kept?.run_id === checkpoint?.iteration_id;

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <h2 className="text-xl font-semibold text-stone-950">Latest kept iteration</h2>
      {kept ? (
        <div className="mt-3 space-y-3 text-sm leading-6 text-stone-700">
          {isPoweringDemo ? (
            <p className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 font-medium text-emerald-950">
              Latest kept model: {kept.run_id}. This model is powering the Doctor and Patient demo.
            </p>
          ) : (
            <p className="rounded-md border border-sky-200 bg-sky-50 px-3 py-2 font-medium text-sky-950">
              Latest kept model: {kept.run_id}. It passed the locked safety evaluation and is ready for
              manual promotion.
            </p>
          )}
          <p>
            Locked evaluation mAP:{" "}
            <span className="font-semibold text-stone-950">
              {formatNumber(kept.locked_eval_scored_map50_95)}
            </span>
          </p>
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

function BadgeChip({ badge }: { badge: HayashiBadge }) {
  return (
    <span className={`inline-flex rounded-md border px-3 py-1 text-sm font-semibold ${badgeClasses[badge]}`}>
      {badge.replaceAll("_", " ")}
    </span>
  );
}

function checkpointLabel(option: InferenceCheckpointOption) {
  const reason = option.discard_reason || option.unusable_reason;
  return [
    option.run_id,
    option.decision,
    `locked_eval ${formatNumber(option.locked_eval_scored_map50_95)}`,
    `research_val ${formatNumber(option.research_val_scored_map50_95)}`,
    reason ? reason : null,
  ]
    .filter(Boolean)
    .join(" | ");
}

function CheckpointSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: InferenceCheckpointOption[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-sm font-semibold text-stone-800">{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-2 w-full rounded-md border border-stone-300 bg-white px-3 py-2 text-sm text-stone-900"
      >
        <option value="">Select an iteration</option>
        {options.map((option) => (
          <option key={option.run_id} value={option.run_id} disabled={!option.usable}>
            {checkpointLabel(option)}
          </option>
        ))}
      </select>
    </label>
  );
}

function CheckpointNote({ option }: { option: InferenceCheckpointOption | undefined }) {
  if (!option) {
    return null;
  }
  if (!option.usable) {
    return (
      <p className="rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm font-medium text-stone-900">
        {option.run_id} cannot run inference: {option.unusable_reason ?? "not usable"}.
      </p>
    );
  }
  if (option.decision !== "KEEP") {
    const reason = option.discard_reason ? ` Keep rule reason: ${option.discard_reason}.` : "";
    return (
      <p className="rounded-md border border-sky-200 bg-sky-50 px-3 py-2 text-sm font-medium text-sky-950">
        {option.run_id} is exploratory/demo only.{reason} This does not change the active checkpoint.
      </p>
    );
  }
  return (
    <p className="rounded-md border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm font-medium text-emerald-950">
      {option.run_id} is an official kept iteration for the research story.
    </p>
  );
}

function CountsSummary({ response }: { response: InferenceResponse }) {
  return (
    <div className="overflow-hidden rounded-md border border-stone-200">
      <table className="w-full border-collapse text-left text-sm">
        <thead className="bg-stone-100 text-stone-700">
          <tr>
            <th className="px-3 py-2 font-semibold">Class</th>
            <th className="px-3 py-2 text-right font-semibold">Count</th>
          </tr>
        </thead>
        <tbody>
          {SCORED_CLASSES.map((className: ScoredClass) => (
            <tr key={className} className="border-t border-stone-200">
              <td className="px-3 py-2 text-stone-800">{classLabels[className]}</td>
              <td className="px-3 py-2 text-right font-semibold text-stone-950">
                {response.counts[className]}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="border-t border-stone-200 px-3 py-2 text-sm text-stone-500">
        Post-acne marks: {response.counts.post_acne_mark}
      </p>
    </div>
  );
}

function CompareResultPanel({
  title,
  option,
  selection,
  response,
}: {
  title: string;
  option: InferenceCheckpointOption;
  selection: ImageSelection;
  response: InferenceResponse;
}) {
  return (
    <section className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-lg font-semibold text-stone-950">{title}</h3>
          <p className="mt-1 text-sm text-stone-600">
            {option.run_id} | {option.decision} | {option.preprocessing}
          </p>
        </div>
        <BadgeChip badge={response.hayashi_badge} />
      </div>
      <OverlayCanvas imageUrl={selection.url} detections={response.detections} />
      <CountsSummary response={response} />
    </section>
  );
}

function DeltaPanel({ result }: { result: IterationCompareResponse }) {
  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="text-lg font-semibold text-stone-950">B minus A count delta</h3>
        <div className="flex items-center gap-2 text-sm text-stone-700">
          <BadgeChip badge={result.deltas.badge_change.from} />
          <span>to</span>
          <BadgeChip badge={result.deltas.badge_change.to} />
        </div>
      </div>
      <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {SCORED_CLASSES.map((className) => {
          const value = result.deltas.counts_diff[className];
          const direction = value > 0 ? "+" : "";
          return (
            <div key={className} className="rounded-md border border-stone-200 bg-stone-50 px-3 py-2">
              <p className="text-sm text-stone-600">{classLabels[className]}</p>
              <p className="mt-1 text-xl font-semibold text-stone-950">
                {direction}
                {value}
              </p>
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-sm text-stone-500">
        Post-acne marks: {result.deltas.counts_diff.post_acne_mark >= 0 ? "+" : ""}
        {result.deltas.counts_diff.post_acne_mark}
      </p>
    </section>
  );
}

function ComparePanel({
  checkpoint,
  options,
  demoPhotos,
}: {
  checkpoint: ActiveCheckpoint | null;
  options: InferenceCheckpointOption[];
  demoPhotos: DemoPhoto[];
}) {
  const [iterationA, setIterationA] = useState("");
  const [iterationB, setIterationB] = useState("");
  const [selection, setSelection] = useState<ImageSelection | null>(null);
  const [result, setResult] = useState<IterationCompareResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [sampleLoadingUrl, setSampleLoadingUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selectionRef = useRef<ImageSelection | null>(null);

  const selectedA = useMemo(
    () => options.find((option) => option.run_id === iterationA),
    [iterationA, options],
  );
  const selectedB = useMemo(
    () => options.find((option) => option.run_id === iterationB),
    [iterationB, options],
  );
  const usableCount = useMemo(() => usableOptions(options).length, [options]);

  useEffect(() => {
    selectionRef.current = selection;
  }, [selection]);

  useEffect(() => () => revokeSelection(selectionRef.current), []);

  useEffect(() => {
    const validCurrent = options.some((option) => option.run_id === iterationA && option.usable);
    if (!validCurrent) {
      setIterationA(defaultIterationA(options, checkpoint)?.run_id ?? "");
    }
  }, [checkpoint, iterationA, options]);

  useEffect(() => {
    const validCurrent = options.some(
      (option) => option.run_id === iterationB && option.usable && option.run_id !== iterationA,
    );
    if (!validCurrent) {
      setIterationB(defaultIterationB(options, iterationA)?.run_id ?? "");
    }
  }, [iterationA, iterationB, options]);

  function setPhoto(file: File) {
    setError(null);
    setResult(null);
    setSelection((previous) => {
      revokeSelection(previous);
      return { file, url: makeObjectUrl(file) };
    });
  }

  async function selectSample(photo: DemoPhoto) {
    setSampleLoadingUrl(photo.url);
    setError(null);
    try {
      setPhoto(await fileFromDemoPhoto(photo));
    } catch (error) {
      setError(error instanceof Error ? error.message : "Unable to load sample photo.");
    } finally {
      setSampleLoadingUrl(null);
    }
  }

  useEffect(() => {
    if (!selection || !selectedA?.usable || !selectedB?.usable || iterationA === iterationB) {
      return;
    }

    let cancelled = false;
    setLoading(true);
    setError(null);
    void compareIterations(selection.file, iterationA, iterationB)
      .then((nextResult) => {
        if (!cancelled) {
          setResult(nextResult);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setResult(null);
          setError(error instanceof Error ? error.message : "Iteration comparison failed.");
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
  }, [iterationA, iterationB, selectedA?.usable, selectedB?.usable, selection]);

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-stone-950">Annotation compare</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-stone-700">
            Pick two inference-ready iterations and run them on the same photo. Discarded checkpoints
            stay exploratory/demo only; this panel is for visual comparison and does not change the demo
            checkpoint.
          </p>
        </div>
        <p className="rounded-md border border-stone-200 bg-stone-50 px-3 py-2 text-sm font-medium text-stone-700">
          {usableCount} inference-ready iteration{usableCount === 1 ? "" : "s"}
        </p>
      </div>

      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        <CheckpointSelect label="Iteration A" value={iterationA} options={options} onChange={setIterationA} />
        <CheckpointSelect label="Iteration B" value={iterationB} options={options} onChange={setIterationB} />
      </div>

      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        <CheckpointNote option={selectedA} />
        <CheckpointNote option={selectedB} />
      </div>

      {iterationA && iterationA === iterationB ? (
        <p className="mt-4 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm font-medium text-stone-900">
          Select two different iterations to compare annotations.
        </p>
      ) : null}

      <div className="mt-5 rounded-md border border-dashed border-stone-300 bg-stone-50 px-4 py-4">
        <label className="block">
          <span className="text-sm font-semibold text-stone-800">Upload one photo</span>
          <input
            type="file"
            accept="image/*"
            onChange={(event) => {
              const [file] = Array.from(event.target.files ?? []);
              if (file) {
                setPhoto(file);
              }
            }}
            className="mt-3 block w-full text-sm text-stone-700 file:mr-4 file:rounded-md file:border file:border-stone-300 file:bg-white file:px-3 file:py-2 file:text-sm file:font-semibold file:text-stone-800 hover:file:border-emerald-600"
          />
        </label>

        {demoPhotos.length > 0 ? (
          <div className="mt-4">
            <p className="text-sm font-semibold text-stone-700">Try a sample</p>
            <div className="mt-2 flex flex-wrap gap-2">
              {demoPhotos.map((photo) => (
                <button
                  key={photo.url}
                  type="button"
                  onClick={() => void selectSample(photo)}
                  className="rounded-md border border-stone-300 bg-white px-3 py-2 text-sm font-medium text-stone-800 hover:border-emerald-600 hover:text-emerald-800 disabled:cursor-wait disabled:opacity-70"
                  disabled={sampleLoadingUrl === photo.url}
                >
                  {sampleLoadingUrl === photo.url ? "Loading..." : photo.label}
                </button>
              ))}
            </div>
          </div>
        ) : null}
      </div>

      {loading ? (
        <div className="mt-4 flex items-center gap-3 rounded-md border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-medium text-sky-950">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-sky-300 border-t-sky-800" />
          Comparing annotations...
        </div>
      ) : null}
      {error ? (
        <div className="mt-4 rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm font-medium text-stone-900">
          {error}
        </div>
      ) : null}
      {!selection ? (
        <p className="mt-4 rounded-md border border-stone-200 bg-stone-50 px-4 py-3 text-sm text-stone-600">
          Add one photo to compare both selected iterations on the same image.
        </p>
      ) : null}

      {selection && result ? (
        <div className="mt-5 space-y-5">
          <div className="grid gap-5 lg:grid-cols-2">
            <CompareResultPanel
              title="Iteration A overlay"
              option={result.iteration_a}
              selection={selection}
              response={result.a}
            />
            <CompareResultPanel
              title="Iteration B overlay"
              option={result.iteration_b}
              selection={selection}
              response={result.b}
            />
          </div>
          <DeltaPanel result={result} />
        </div>
      ) : null}
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
    { key: "run_id", label: "Run" },
    { key: "timestamp", label: "Time" },
    { key: "decision", label: "Decision" },
    { key: "discard_reason", label: "Reason" },
    { key: "research_val_scored_map50_95", label: "Research mAP", numeric: true },
    { key: "locked_eval_scored_map50_95", label: "Eval mAP", numeric: true },
    { key: "locked_eval_nodule_cyst_recall", label: "Nodule Recall", numeric: true },
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

function demoPhotoLabel(fileName: string, index: number) {
  const labels: Record<string, string> = {
    "fresh_single_01.png": "Fresh sample 1",
    "fresh_single_02.png": "Fresh sample 2",
    "fresh_pair_01_before.png": "Pair 1 before",
    "fresh_pair_01_after.png": "Pair 1 after",
    "fresh_pair_02_before.png": "Pair 2 before",
    "fresh_pair_02_after.png": "Pair 2 after",
    "fresh_pair_03_view_a.png": "Pair 3 view A",
    "fresh_pair_03_view_b.png": "Pair 3 view B",
    "fresh_pair_04_before.png": "Pair 4 before",
    "fresh_pair_04_after.png": "Pair 4 after",
    "fresh_pair_05_before.png": "Pair 5 before",
    "fresh_pair_05_after.png": "Pair 5 after",
  };
  return labels[fileName] ?? `Sample ${index + 1}`;
}

export const getStaticProps: GetStaticProps<ResearchPageProps> = async () => {
  const fs = await import("fs");
  const path = await import("path");
  const cwd = process.cwd();
  const demoDirCandidates = [
    path.join(cwd, "public", "demo_photos"),
    path.join(cwd, "web", "public", "demo_photos"),
  ];
  const demoDir = demoDirCandidates.find((candidate) => fs.existsSync(candidate));
  const imageExtensions = new Set([".jpg", ".jpeg", ".png", ".webp"]);
  const demoPhotos = demoDir
    ? fs
        .readdirSync(demoDir)
        .filter((fileName) => imageExtensions.has(path.extname(fileName).toLowerCase()))
        .sort()
        .map((fileName, index) => ({
          label: demoPhotoLabel(fileName, index),
          url: `/demo_photos/${fileName}`,
        }))
    : [];

  return { props: { demoPhotos } };
};

export default function ResearchPage({ demoPhotos }: ResearchPageProps) {
  const [checkpoint, setCheckpoint] = useState<ActiveCheckpoint | null>(null);
  const [rows, setRows] = useState<ExperimentRow[]>([]);
  const [checkpointOptions, setCheckpointOptions] = useState<InferenceCheckpointOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [nextCheckpoint, nextRows, nextCheckpointOptions] = await Promise.all([
          fetchJson<ActiveCheckpoint>("/api/active_checkpoint"),
          fetchJson<ExperimentRow[]>("/api/iterations"),
          fetchJson<InferenceCheckpointOption[]>("/api/inference_checkpoints"),
        ]);
        if (!cancelled) {
          setCheckpoint(nextCheckpoint);
          setRows(nextRows);
          setCheckpointOptions(nextCheckpointOptions);
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
    <Layout activeTab="research" pageTitle="AutoDerm — Research">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-semibold tracking-normal text-stone-950">
            Research — Autoresearch Loop Status
          </h1>
          <p className="mt-3 max-w-4xl text-base leading-7 text-stone-700">
            Codex proposes bounded changes to the training code, runs experiments, and keeps only changes that
            improve locked evaluation while preserving clinical safety guardrails. Each dot below is one
            experiment: green was kept, gray was discarded, red failed before evaluation.
          </p>
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
        <ComparePanel checkpoint={checkpoint} options={checkpointOptions} demoPhotos={demoPhotos} />
        <ResearchScatter rows={rows} />
        <IterationsTable rows={rows} />
      </div>
    </Layout>
  );
}
