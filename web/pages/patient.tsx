import type { GetStaticProps } from "next";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDropzone } from "react-dropzone";

import Layout from "../components/Layout";
import OverlayCanvas from "../components/OverlayCanvas";
import OverlayLegend from "../components/OverlayLegend";
import type { AnyClass, HayashiBadge, InferenceResponse } from "../lib/types";

type DemoPhoto = {
  label: string;
  url: string;
};

type ImageSelection = {
  file: File;
  url: string;
};

type PatientPageProps = {
  demoPhotos: DemoPhoto[];
};

type LesionCard = {
  key: string;
  title: string;
  count: number;
  body: string;
};

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

const badgeClasses: Record<HayashiBadge, string> = {
  clear: "border-emerald-300 bg-emerald-50 text-emerald-900",
  almost_clear: "border-lime-300 bg-lime-50 text-lime-900",
  mild: "border-yellow-300 bg-yellow-50 text-yellow-900",
  moderate: "border-rose-300 bg-rose-50 text-rose-900",
  severe: "border-red-300 bg-red-50 text-red-900",
};

function inflammatoryCount(counts: InferenceResponse["counts"]) {
  return counts.papule + counts.pustule + counts.nodule_cyst;
}

function comedoneCount(counts: InferenceResponse["counts"]) {
  return counts.comedone_open + counts.comedone_closed;
}

function badgeLabel(badge: HayashiBadge) {
  return badge.replaceAll("_", " ");
}

function makeObjectUrl(file: File) {
  return URL.createObjectURL(file);
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

async function fileFromDemoPhoto(photo: DemoPhoto) {
  const response = await fetch(photo.url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Unable to load sample photo.");
  }
  const blob = await response.blob();
  return new File([blob], photo.label, { type: blob.type || "image/jpeg" });
}

async function inferPatientPhoto(file: File) {
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

function UploadArea({ onSelect }: { onSelect: (file: File) => void }) {
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
        "autoderm-dropzone flex min-h-52 cursor-pointer flex-col items-center justify-center rounded-md border px-5 py-8 text-center transition",
        isDragActive
          ? "border-emerald-500 bg-emerald-50"
          : "border-stone-300 bg-white hover:border-stone-400",
      ].join(" ")}
    >
      <input {...getInputProps()} />
      <span className="autoderm-upload-mark mb-4" aria-hidden="true">
        +
      </span>
      <p className="text-base font-semibold text-stone-950">Upload one clinical photo</p>
      <p className="mt-2 max-w-sm text-sm text-stone-600">
        Drag a photo here, or click to choose an image.
      </p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="flex items-center gap-3 rounded-md border border-sky-200 bg-sky-50 px-4 py-3 text-sm font-medium text-sky-950">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-sky-300 border-t-sky-800" />
      Reading the photo...
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
        <div className="flex gap-3 overflow-x-auto pb-2">
          {demoPhotos.map((photo) => (
            <button
              key={photo.url}
              type="button"
              onClick={() => void selectSample(photo)}
              className="autoderm-sample-tile group w-36 shrink-0 rounded-md border border-stone-300 bg-white p-2 text-left text-sm font-medium text-stone-800 hover:border-emerald-600 hover:text-emerald-800 disabled:cursor-wait disabled:opacity-70"
              disabled={loadingUrl === photo.url}
            >
              <span className="block aspect-[4/3] overflow-hidden rounded-md bg-stone-100">
                <img
                  src={photo.url}
                  alt=""
                  className="h-full w-full object-cover transition duration-300 group-hover:scale-[1.04]"
                />
              </span>
              <span className="mt-2 block truncate">
                {loadingUrl === photo.url ? "Loading..." : photo.label}
              </span>
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

function BadgeChip({ badge }: { badge: HayashiBadge }) {
  return (
    <span className={`inline-flex rounded-md border px-3 py-1 text-sm font-semibold ${badgeClasses[badge]}`}>
      {badgeLabel(badge)}
    </span>
  );
}

function WhatWeFound({ response }: { response: InferenceResponse }) {
  const activeInflammatory = inflammatoryCount(response.counts);

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <h2 className="text-xl font-semibold text-stone-950">What we found</h2>
      <div className="mt-4 space-y-3 text-base leading-7 text-stone-800">
        <p>We detected {activeInflammatory} active inflammatory acne lesions on this photo.</p>
        <p>
          Severity: <BadgeChip badge={response.hayashi_badge} />
        </p>
        <p>
          We do NOT detect: dark spots, scarring, or post-acne marks. Those need separate clinical evaluation.
        </p>
      </div>
    </section>
  );
}

function lesionCards(counts: InferenceResponse["counts"]): LesionCard[] {
  const cards: LesionCard[] = [];
  const totalComedones = comedoneCount(counts);
  if (totalComedones > 0) {
    cards.push({
      key: "comedones",
      title: "Comedones",
      count: totalComedones,
      body: "Blocked pores. Open comedones are blackheads and closed comedones are whiteheads. Not inflamed.",
    });
  }
  if (counts.papule > 0) {
    cards.push({
      key: "papules",
      title: "Papules",
      count: counts.papule,
      body: "Small red bumps. Inflamed. No pus.",
    });
  }
  if (counts.pustule > 0) {
    cards.push({
      key: "pustules",
      title: "Pustules",
      count: counts.pustule,
      body: "Red bumps with visible pus.",
    });
  }
  if (counts.nodule_cyst > 0) {
    cards.push({
      key: "nodule_cyst",
      title: "Nodules/cysts",
      count: counts.nodule_cyst,
      body: "Large, deep, painful. Most likely to scar.",
    });
  }
  return cards;
}

function LesionTypesExplained({ response }: { response: InferenceResponse }) {
  const cards = lesionCards(response.counts);

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <h2 className="text-xl font-semibold text-stone-950">Lesion types explained</h2>
      {cards.length > 0 ? (
        <div className="mt-4 space-y-3">
          {cards.map((card) => (
            <details key={card.key} className="rounded-md border border-stone-200 bg-stone-50 px-4 py-3">
              <summary className="cursor-pointer text-base font-semibold text-stone-950">
                {card.title}: {card.count}
              </summary>
              <p className="mt-3 text-sm leading-6 text-stone-700">{card.body}</p>
            </details>
          ))}
        </div>
      ) : (
        <p className="mt-4 text-sm leading-6 text-stone-700">
          No explained acne lesion types were detected in this photo.
        </p>
      )}
    </section>
  );
}

function careGuidance(response: InferenceResponse) {
  const counts = response.counts;
  const activeInflammatory = inflammatoryCount(counts);
  const totalComedones = comedoneCount(counts);
  const recommendations: string[] = [];

  if (totalComedones > 3 && activeInflammatory < 5) {
    recommendations.push("Salicylic acid cleanser + adapalene 0.1% gel at night.");
  }
  if (activeInflammatory >= 5 && activeInflammatory <= 15) {
    recommendations.push("Benzoyl peroxide 2.5% wash + adapalene 0.1% at night + niacinamide 5% serum.");
  }
  if (activeInflammatory > 15 || counts.nodule_cyst > 2) {
    recommendations.push(
      "These findings warrant a dermatologist consultation. Over-the-counter regimens are unlikely to be sufficient.",
    );
  }
  recommendations.push("Use sunscreen daily.");
  return recommendations;
}

function CareGuidance({ response }: { response: InferenceResponse }) {
  return (
    <section className="rounded-md border border-stone-200 bg-white p-4">
      <h2 className="text-xl font-semibold text-stone-950">Care guidance</h2>
      <ul className="mt-4 space-y-2 text-base leading-7 text-stone-800">
        {careGuidance(response).map((item) => (
          <li key={item} className="rounded-md bg-stone-50 px-3 py-2">
            {item}
          </li>
        ))}
      </ul>
    </section>
  );
}

function DermatologistGuidance({ response }: { response: InferenceResponse }) {
  const hasNodule = response.counts.nodule_cyst > 0;

  return (
    <section
      className={[
        "rounded-md border p-4",
        hasNodule ? "border-red-400 bg-red-50" : "border-stone-200 bg-white",
      ].join(" ")}
    >
      <h2 className="text-xl font-semibold text-stone-950">When to see a dermatologist</h2>
      <p className="mt-4 text-base leading-7 text-stone-800">
        If you have nodules or cysts, persistent acne despite OTC treatment for 8 weeks, or scarring, see a
        dermatologist.
      </p>
    </section>
  );
}

function ResultDetails({ response }: { response: InferenceResponse }) {
  return (
    <div className="grid gap-4 lg:grid-cols-3">
      <LesionTypesExplained response={response} />
      <CareGuidance response={response} />
      <DermatologistGuidance response={response} />
      <p className="rounded-md border border-amber-300 bg-amber-50 px-5 py-4 text-base font-semibold leading-7 text-stone-950 lg:col-span-3">
        Research demonstration, not a medical diagnosis. Educational only. Not a substitute for evaluation by a
        qualified dermatologist.
      </p>
    </div>
  );
}

export const getStaticProps: GetStaticProps<PatientPageProps> = async () => {
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

export default function PatientPage({ demoPhotos }: PatientPageProps) {
  const [selection, setSelection] = useState<ImageSelection | null>(null);
  const [response, setResponse] = useState<InferenceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const selectionRef = useRef<ImageSelection | null>(null);

  const selectFile = useCallback((file: File) => {
    setError(null);
    setResponse(null);
    setSelection((previous) => {
      revokeSelection(previous);
      return { file, url: makeObjectUrl(file) };
    });
  }, []);

  useEffect(() => {
    selectionRef.current = selection;
  }, [selection]);

  useEffect(() => () => revokeSelection(selectionRef.current), []);

  useEffect(() => {
    if (!selection) {
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    void inferPatientPhoto(selection.file)
      .then((nextResponse) => {
        if (!cancelled) {
          setResponse(nextResponse);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setResponse(null);
          setError(error instanceof Error ? error.message : "Inference failed.");
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
  }, [selection]);

  const hasResult = useMemo(() => Boolean(selection && response), [response, selection]);

  return (
    <Layout activeTab="patient" pageTitle="AutoDerm — Patient">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-semibold tracking-normal text-stone-950">
            Patient — What This Means For You
          </h1>
        </div>

        <div className="space-y-5">
          <UploadArea onSelect={selectFile} />
          <SampleButtons demoPhotos={demoPhotos} onSelect={selectFile} />
          {loading ? <LoadingState /> : null}
          {error ? <ErrorState message={error} /> : null}
        </div>

        {hasResult && selection && response ? (
          <div className="space-y-5">
            <div className="grid gap-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(300px,0.65fr)] lg:items-start">
              <div className="space-y-3">
                <OverlayCanvas imageUrl={selection.url} detections={response.detections} labelMode="patient" />
                <OverlayLegend mode="patient" />
              </div>
              <WhatWeFound response={response} />
            </div>
            <ResultDetails response={response} />
          </div>
        ) : null}
      </div>
    </Layout>
  );
}
