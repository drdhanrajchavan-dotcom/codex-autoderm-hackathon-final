import type { GetStaticProps } from "next";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";

import Layout from "../components/Layout";
import OverlayCanvas from "../components/OverlayCanvas";
import OverlayLegend from "../components/OverlayLegend";
import type { AnyClass, HayashiBadge, InferenceResponse, ScoredClass } from "../lib/types";
import { SCORED_CLASSES } from "../lib/types";

type Mode = "single" | "pair";

type DemoPhoto = {
  label: string;
  url: string;
};

type PairResponse = {
  before: InferenceResponse;
  after: InferenceResponse;
  deltas: {
    counts_diff: Record<AnyClass, number>;
    badge_change: {
      from: HayashiBadge;
      to: HayashiBadge;
    };
  };
};

type ImageSelection = {
  file: File;
  url: string;
};

type DoctorPageProps = {
  demoPhotos: DemoPhoto[];
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

const badgeClasses: Record<HayashiBadge, string> = {
  clear: "border-emerald-300 bg-emerald-50 text-emerald-900",
  almost_clear: "border-lime-300 bg-lime-50 text-lime-900",
  mild: "border-yellow-300 bg-yellow-50 text-yellow-900",
  moderate: "border-rose-300 bg-rose-50 text-rose-900",
  severe: "border-red-300 bg-red-50 text-red-900",
};

const classLabels: Record<AnyClass, string> = {
  comedone_open: "Open comedones",
  comedone_closed: "Closed comedones",
  papule: "Papules",
  pustule: "Pustules",
  nodule_cyst: "Nodules or cysts",
  post_acne_mark: "Post-acne marks",
};

const demoPairs = [
  {
    label: "Pair 1",
    before: "fresh_pair_01_before.png",
    after: "fresh_pair_01_after.png",
  },
  {
    label: "Pair 2",
    before: "fresh_pair_02_before.png",
    after: "fresh_pair_02_after.png",
  },
  {
    label: "Pair 4",
    before: "fresh_pair_04_before.png",
    after: "fresh_pair_04_after.png",
  },
  {
    label: "Pair 5",
    before: "fresh_pair_05_before.png",
    after: "fresh_pair_05_after.png",
  },
];

function activeLesionCount(counts: InferenceResponse["counts"]) {
  return counts.papule + counts.pustule + counts.nodule_cyst;
}

function dominantClass(counts: InferenceResponse["counts"]) {
  const ranked = [...SCORED_CLASSES].sort((a, b) => counts[b] - counts[a]);
  const top = ranked[0];
  return counts[top] > 0 ? classLabels[top].toLowerCase() : "no scored lesion class";
}

function clinicalObservation(response: InferenceResponse) {
  const inflammatory = activeLesionCount(response.counts);
  const noduleCount = response.counts.nodule_cyst;
  const lines = [
    `${response.hayashi_badge.replaceAll("_", " ")} inflammatory acne, ${inflammatory} active lesions detected. Predominantly ${dominantClass(response.counts)}.`,
  ];
  if (noduleCount > 0) {
    lines.push(`${noduleCount} nodule(s) detected - clinically significant.`);
  }
  return lines;
}

function deltaTextClass(value: number) {
  if (value < 0) {
    return "text-emerald-700";
  }
  if (value > 0) {
    return "text-red-700";
  }
  return "text-stone-950";
}

function deltaCardClass(value: number) {
  if (value < 0) {
    return "border-emerald-200 bg-emerald-50";
  }
  if (value > 0) {
    return "border-red-200 bg-red-50";
  }
  return "border-stone-200 bg-stone-50";
}

function deltaPercent(value: number, beforeCount: number) {
  if (beforeCount === 0) {
    return null;
  }
  return Math.round((value / beforeCount) * 100);
}

function makeObjectUrl(file: File) {
  return URL.createObjectURL(file);
}

async function fileFromDemoPhoto(photo: DemoPhoto) {
  const response = await fetch(photo.url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Unable to load sample photo.");
  }
  const blob = await response.blob();
  return new File([blob], photo.label, { type: blob.type || "image/jpeg" });
}

async function inferSingle(file: File) {
  const body = new FormData();
  body.append("image", file);
  const response = await fetch(`${API_BASE}/api/infer`, {
    method: "POST",
    body,
    cache: "no-store",
  });
  if (response.status === 503) {
    throw new Error("Baselines still running. Check Research tab for status.");
  }
  if (!response.ok) {
    throw new Error("Inference failed. Check the API server and active checkpoint.");
  }
  return (await response.json()) as InferenceResponse;
}

async function inferPair(before: File, after: File) {
  const body = new FormData();
  body.append("before", before);
  body.append("after", after);
  const response = await fetch(`${API_BASE}/api/infer_pair`, {
    method: "POST",
    body,
    cache: "no-store",
  });
  if (response.status === 503) {
    throw new Error("Baselines still running. Check Research tab for status.");
  }
  if (!response.ok) {
    throw new Error("Pair inference failed. Check the API server and active checkpoint.");
  }
  return (await response.json()) as PairResponse;
}

function UploadArea({
  label,
  onSelect,
}: {
  label: string;
  onSelect: (file: File) => void;
}) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const [file] = acceptedFiles;
      if (file) {
        onSelect(file);
      }
    },
    [onSelect],
  );
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: { "image/*": [] },
    maxFiles: 1,
    onDrop,
  });

  return (
    <div
      {...getRootProps()}
      className={[
        "flex min-h-44 cursor-pointer flex-col items-center justify-center rounded-md border border-dashed px-4 py-6 text-center transition",
        isDragActive
          ? "border-emerald-600 bg-emerald-50"
          : "border-stone-300 bg-white hover:border-rose-300 hover:bg-rose-50",
      ].join(" ")}
    >
      <input {...getInputProps()} />
      <p className="text-base font-semibold text-stone-950">{label}</p>
      <p className="mt-2 max-w-sm text-sm text-stone-600">
        Drag a clinical photo here, or click to choose an image.
      </p>
    </div>
  );
}

function LoadingState({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-medium text-sky-950">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-sky-300 border-t-sky-800" />
      {label}
    </div>
  );
}

function ErrorState({ message }: { message: string }) {
  return (
    <div className="rounded-md border border-amber-300 bg-amber-50 px-4 py-3 text-sm font-medium text-stone-900">
      {message}
    </div>
  );
}

function BadgeChip({ badge }: { badge: HayashiBadge }) {
  return (
    <span className={`inline-flex rounded-md border px-3 py-1 text-sm font-semibold ${badgeClasses[badge]}`}>
      {badge.replaceAll("_", " ")}
    </span>
  );
}

function CountsPanel({ response }: { response: InferenceResponse }) {
  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-medium text-stone-600">Inflammatory total</p>
          <p className="mt-1 text-3xl font-semibold text-stone-950">{activeLesionCount(response.counts)}</p>
        </div>
        <BadgeChip badge={response.hayashi_badge} />
      </div>
      <div className="mt-4 overflow-hidden rounded-md border border-stone-200">
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
                <td className="px-3 py-2 text-right font-semibold text-stone-950">{response.counts[className]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-3 text-sm text-stone-500">
        Post-acne marks: {response.counts.post_acne_mark}
      </p>
    </section>
  );
}

function ClinicalObservation({ response }: { response: InferenceResponse }) {
  return (
    <section className="rounded-md border border-emerald-200 bg-emerald-50 p-4">
      <h2 className="text-base font-semibold text-stone-950">Clinical observation</h2>
      <div className="mt-3 space-y-2 text-sm leading-6 text-stone-800">
        {clinicalObservation(response).map((line) => (
          <p key={line}>{line}</p>
        ))}
      </div>
    </section>
  );
}

function SampleButtons({
  demoPhotos,
  onSelect,
}: {
  demoPhotos: DemoPhoto[];
  onSelect: (file: File) => void;
}) {
  const [loadingUrl, setLoadingUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function selectSample(photo: DemoPhoto) {
    setError(null);
    setLoadingUrl(photo.url);
    try {
      onSelect(await fileFromDemoPhoto(photo));
    } catch (error) {
      setError(error instanceof Error ? error.message : "Unable to load sample photo.");
    } finally {
      setLoadingUrl(null);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-sm font-semibold text-stone-700">Try a sample</p>
      {demoPhotos.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {demoPhotos.map((photo) => (
            <button
              key={photo.url}
              type="button"
              onClick={() => void selectSample(photo)}
              className="rounded-md border border-stone-300 bg-white px-3 py-2 text-sm font-medium text-stone-800 hover:border-emerald-600 hover:text-emerald-800 disabled:cursor-wait disabled:opacity-70"
              disabled={loadingUrl === photo.url}
            >
              {loadingUrl === photo.url ? "Loading..." : photo.label}
            </button>
          ))}
        </div>
      ) : (
        <p className="rounded-md border border-stone-200 bg-stone-50 px-3 py-2 text-sm text-stone-600">
          Sample photos will appear here when fresh demo images are added.
        </p>
      )}
      {error ? <ErrorState message={error} /> : null}
    </div>
  );
}

function fileNameFromUrl(url: string) {
  return decodeURIComponent(url.split("/").pop() ?? url);
}

function PairSampleButtons({
  demoPhotos,
  onSelectBefore,
  onSelectAfter,
}: {
  demoPhotos: DemoPhoto[];
  onSelectBefore: (file: File) => void;
  onSelectAfter: (file: File) => void;
}) {
  const [loadingLabel, setLoadingLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const photosByName = useMemo(
    () => new Map(demoPhotos.map((photo) => [fileNameFromUrl(photo.url), photo])),
    [demoPhotos],
  );
  const availablePairs = useMemo(
    () =>
      demoPairs
        .map((pair) => ({
          ...pair,
          beforePhoto: photosByName.get(pair.before),
          afterPhoto: photosByName.get(pair.after),
        }))
        .filter((pair) => pair.beforePhoto && pair.afterPhoto),
    [photosByName],
  );

  async function selectPair(pair: (typeof availablePairs)[number]) {
    if (!pair.beforePhoto || !pair.afterPhoto) {
      return;
    }
    setError(null);
    setLoadingLabel(pair.label);
    try {
      const [before, after] = await Promise.all([
        fileFromDemoPhoto(pair.beforePhoto),
        fileFromDemoPhoto(pair.afterPhoto),
      ]);
      onSelectBefore(before);
      onSelectAfter(after);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Unable to load demo pair.");
    } finally {
      setLoadingLabel(null);
    }
  }

  return (
    <div className="space-y-3">
      <p className="text-sm font-semibold text-stone-700">Load a demo pair</p>
      {availablePairs.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {availablePairs.map((pair) => (
            <button
              key={pair.label}
              type="button"
              onClick={() => void selectPair(pair)}
              className="rounded-md border border-stone-300 bg-white px-3 py-2 text-sm font-medium text-stone-800 hover:border-emerald-600 hover:text-emerald-800 disabled:cursor-wait disabled:opacity-70"
              disabled={loadingLabel === pair.label}
            >
              {loadingLabel === pair.label ? "Loading..." : pair.label}
            </button>
          ))}
        </div>
      ) : (
        <p className="rounded-md border border-stone-200 bg-stone-50 px-3 py-2 text-sm text-stone-600">
          Demo pairs will appear here when before/after samples are available.
        </p>
      )}
      {error ? <ErrorState message={error} /> : null}
    </div>
  );
}

function OverlayResult({
  title,
  selection,
  response,
}: {
  title?: string;
  selection: ImageSelection;
  response: InferenceResponse;
}) {
  return (
    <section className="space-y-4">
      {title ? <h2 className="text-lg font-semibold text-stone-950">{title}</h2> : null}
      <div className="space-y-3">
        <OverlayCanvas imageUrl={selection.url} detections={response.detections} labelMode="technical" />
        <OverlayLegend mode="doctor" />
      </div>
      <CountsPanel response={response} />
    </section>
  );
}

function DeltaPanel({ pair }: { pair: PairResponse }) {
  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 className="text-lg font-semibold text-stone-950">Count changes</h2>
        <div className="flex items-center gap-2 text-sm text-stone-700">
          <BadgeChip badge={pair.deltas.badge_change.from} />
          <span>to</span>
          <BadgeChip badge={pair.deltas.badge_change.to} />
        </div>
      </div>
      <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {SCORED_CLASSES.map((className) => {
          const value = pair.deltas.counts_diff[className];
          const percent = deltaPercent(value, pair.before.counts[className]);
          const direction = value > 0 ? "+" : "";
          return (
            <div key={className} className={`rounded-md border px-3 py-2 ${deltaCardClass(value)}`}>
              <p className="text-sm text-stone-600">{classLabels[className]}</p>
              <p className={`mt-1 text-xl font-semibold ${deltaTextClass(value)}`}>
                {direction}
                {value}
              </p>
              {percent !== null ? (
                <p className={`mt-1 text-xs font-semibold ${deltaTextClass(value)}`}>
                  {percent > 0 ? "+" : ""}
                  {percent}%
                </p>
              ) : null}
            </div>
          );
        })}
      </div>
      <p className="mt-3 text-sm text-stone-500">
        Post-acne marks: {pair.deltas.counts_diff.post_acne_mark >= 0 ? "+" : ""}
        {pair.deltas.counts_diff.post_acne_mark}
      </p>
    </section>
  );
}

function revokeSelection(selection: ImageSelection | null) {
  if (selection) {
    URL.revokeObjectURL(selection.url);
  }
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

export const getStaticProps: GetStaticProps<DoctorPageProps> = async () => {
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

export default function DoctorPage({ demoPhotos }: DoctorPageProps) {
  const [mode, setMode] = useState<Mode>("single");
  const [singleSelection, setSingleSelection] = useState<ImageSelection | null>(null);
  const [singleResponse, setSingleResponse] = useState<InferenceResponse | null>(null);
  const [beforeSelection, setBeforeSelection] = useState<ImageSelection | null>(null);
  const [afterSelection, setAfterSelection] = useState<ImageSelection | null>(null);
  const [pairResponse, setPairResponse] = useState<PairResponse | null>(null);
  const [loading, setLoading] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const selectionsRef = useRef<Array<ImageSelection | null>>([]);

  const setSingleFile = useCallback((file: File) => {
    setError(null);
    setSingleResponse(null);
    setSingleSelection((previous) => {
      revokeSelection(previous);
      return { file, url: makeObjectUrl(file) };
    });
  }, []);

  const setBeforeFile = useCallback((file: File) => {
    setError(null);
    setPairResponse(null);
    setBeforeSelection((previous) => {
      revokeSelection(previous);
      return { file, url: makeObjectUrl(file) };
    });
  }, []);

  const setAfterFile = useCallback((file: File) => {
    setError(null);
    setPairResponse(null);
    setAfterSelection((previous) => {
      revokeSelection(previous);
      return { file, url: makeObjectUrl(file) };
    });
  }, []);

  useEffect(() => {
    selectionsRef.current = [singleSelection, beforeSelection, afterSelection];
  }, [afterSelection, beforeSelection, singleSelection]);

  useEffect(() => () => selectionsRef.current.forEach(revokeSelection), []);

  useEffect(() => {
    if (!singleSelection || mode !== "single") {
      return;
    }
    let cancelled = false;
    setLoading("single");
    setError(null);
    void inferSingle(singleSelection.file)
      .then((response) => {
        if (!cancelled) {
          setSingleResponse(response);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setSingleResponse(null);
          setError(error instanceof Error ? error.message : "Inference failed.");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [mode, singleSelection]);

  useEffect(() => {
    if (!beforeSelection || !afterSelection || mode !== "pair") {
      return;
    }
    let cancelled = false;
    setLoading("pair");
    setError(null);
    void inferPair(beforeSelection.file, afterSelection.file)
      .then((response) => {
        if (!cancelled) {
          setPairResponse(response);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setPairResponse(null);
          setError(error instanceof Error ? error.message : "Pair inference failed.");
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoading(null);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [afterSelection, beforeSelection, mode]);

  const pairReady = useMemo(() => Boolean(beforeSelection && afterSelection), [afterSelection, beforeSelection]);

  return (
    <Layout activeTab="doctor" pageTitle="AutoDerm — Doctor">
      <div className="space-y-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-normal text-stone-950">
              Doctor — Lesion Detection
            </h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-stone-700">
              Single photo detects lesion burden. Pre/post comparison tracks treatment response across visits.
            </p>
          </div>
          <div className="inline-flex w-fit rounded-md border border-stone-300 bg-white p-1">
            {[
              { id: "single" as const, label: "Single Photo" },
              { id: "pair" as const, label: "Pre/Post Comparison" },
            ].map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => {
                  setMode(item.id);
                  setError(null);
                }}
                className={[
                  "rounded-md px-4 py-2 text-sm font-semibold transition",
                  mode === item.id ? "bg-emerald-700 text-white" : "text-stone-700 hover:bg-rose-100",
                ].join(" ")}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>

        {mode === "single" ? (
          <div className="space-y-5">
            <UploadArea label="Upload one clinical photo" onSelect={setSingleFile} />
            <SampleButtons demoPhotos={demoPhotos} onSelect={setSingleFile} />
            {loading === "single" ? <LoadingState label="Running lesion detection..." /> : null}
            {error ? <ErrorState message={error} /> : null}
            {singleSelection && singleResponse ? (
              <div className="grid gap-5 lg:grid-cols-[minmax(0,1.4fr)_minmax(320px,0.8fr)]">
                <div className="space-y-3">
                  <OverlayCanvas
                    imageUrl={singleSelection.url}
                    detections={singleResponse.detections}
                    labelMode="technical"
                  />
                  <OverlayLegend mode="doctor" />
                </div>
                <div className="space-y-4">
                  <CountsPanel response={singleResponse} />
                  <ClinicalObservation response={singleResponse} />
                </div>
              </div>
            ) : null}
          </div>
        ) : (
          <div className="space-y-5">
            <div className="grid gap-4 lg:grid-cols-2">
              <UploadArea label="Upload before photo" onSelect={setBeforeFile} />
              <UploadArea label="Upload after photo" onSelect={setAfterFile} />
            </div>
            <PairSampleButtons
              demoPhotos={demoPhotos}
              onSelectBefore={setBeforeFile}
              onSelectAfter={setAfterFile}
            />
            {loading === "pair" ? <LoadingState label="Comparing both visits..." /> : null}
            {error ? <ErrorState message={error} /> : null}
            {!pairReady ? (
              <p className="rounded-md border border-stone-200 bg-stone-50 px-4 py-3 text-sm text-stone-600">
                Add both visits to run the pre/post comparison.
              </p>
            ) : null}
            {beforeSelection && afterSelection && pairResponse ? (
              <div className="space-y-5">
                <div className="grid gap-5 lg:grid-cols-2">
                  <OverlayResult title="Before" selection={beforeSelection} response={pairResponse.before} />
                  <OverlayResult title="After" selection={afterSelection} response={pairResponse.after} />
                </div>
                <DeltaPanel pair={pairResponse} />
              </div>
            ) : null}
          </div>
        )}
      </div>
    </Layout>
  );
}
