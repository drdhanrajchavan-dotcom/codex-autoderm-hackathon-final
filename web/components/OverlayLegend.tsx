type LegendMode = "doctor" | "patient";

type LegendItem = {
  colors: string[];
  label: string;
};

const doctorItems: LegendItem[] = [
  { colors: ["#f97316"], label: "Open comedone" },
  { colors: ["#eab308"], label: "Closed comedone" },
  { colors: ["#ec4899"], label: "Papule" },
  { colors: ["#dc2626"], label: "Pustule" },
  { colors: ["#7e22ce"], label: "Nodule/cyst" },
  { colors: ["#6b7280"], label: "Post-acne mark" },
];

const patientItems: LegendItem[] = [
  { colors: ["#f97316", "#eab308"], label: "Blocked pore" },
  { colors: ["#ec4899", "#dc2626"], label: "Inflamed spot" },
  { colors: ["#7e22ce"], label: "Deep inflamed spot" },
  { colors: ["#6b7280"], label: "Skin mark" },
];

export default function OverlayLegend({ mode }: { mode: LegendMode }) {
  const items = mode === "patient" ? patientItems : doctorItems;

  return (
    <div className="flex flex-wrap gap-x-4 gap-y-2 rounded-md border border-stone-200 bg-white px-3 py-2 text-sm text-stone-700">
      {items.map((item, index) => (
        <span key={`${item.label}-${index}`} className="inline-flex items-center gap-2">
          <span className="inline-flex items-center -space-x-1" aria-hidden="true">
            {item.colors.map((color) => (
              <span
                key={color}
                className="h-3 w-3 rounded-full border border-white shadow-sm"
                style={{ backgroundColor: color }}
              />
            ))}
          </span>
          {item.label}
        </span>
      ))}
    </div>
  );
}
