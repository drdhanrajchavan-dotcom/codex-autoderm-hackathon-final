import { useEffect, useRef } from "react";

import type { Detection } from "../lib/types";

type OverlayCanvasProps = {
  imageUrl: string;
  detections: Detection[];
  confidenceFloor?: number;
  labelMode?: OverlayLabelMode;
};

export type OverlayLabelMode = "technical" | "patient" | "none";

type Point = {
  x: number;
  y: number;
};

const classColors: Record<Detection["class_name"], { stroke: string; fill: string; alpha: number }> = {
  comedone_open: { stroke: "#f97316", fill: "rgba(249, 115, 22, 0.9)", alpha: 1 },
  comedone_closed: { stroke: "#eab308", fill: "rgba(234, 179, 8, 0.9)", alpha: 1 },
  papule: { stroke: "#ec4899", fill: "rgba(236, 72, 153, 0.9)", alpha: 1 },
  pustule: { stroke: "#dc2626", fill: "rgba(220, 38, 38, 0.9)", alpha: 1 },
  nodule_cyst: { stroke: "#7e22ce", fill: "rgba(126, 34, 206, 0.9)", alpha: 1 },
  post_acne_mark: { stroke: "#6b7280", fill: "rgba(107, 114, 128, 0.7)", alpha: 0.5 },
};

const patientLabels: Record<Detection["class_name"], string> = {
  comedone_open: "blocked pore",
  comedone_closed: "blocked pore",
  papule: "inflamed spot",
  pustule: "inflamed spot",
  nodule_cyst: "deep inflamed spot",
  post_acne_mark: "skin mark",
};

function rotatedBox([x1, y1, x2, y2, angle]: Detection["bbox"]): Point[] {
  const centerX = (x1 + x2) / 2;
  const centerY = (y1 + y2) / 2;
  const halfWidth = Math.max(1, Math.abs(x2 - x1)) / 2;
  const halfHeight = Math.max(1, Math.abs(y2 - y1)) / 2;
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);
  const corners = [
    { x: -halfWidth, y: -halfHeight },
    { x: halfWidth, y: -halfHeight },
    { x: halfWidth, y: halfHeight },
    { x: -halfWidth, y: halfHeight },
  ];

  return corners.map((corner) => ({
    x: centerX + corner.x * cosA - corner.y * sinA,
    y: centerY + corner.x * sinA + corner.y * cosA,
  }));
}

function drawLabel(
  context: CanvasRenderingContext2D,
  text: string,
  point: Point,
  fill: string,
  isAuxiliary: boolean,
) {
  const fontSize = isAuxiliary ? 11 : 12;
  context.font = `${fontSize}px ui-sans-serif, system-ui, sans-serif`;
  const metrics = context.measureText(text);
  const paddingX = 5;
  const paddingY = 4;
  const x = Math.max(0, point.x);
  const y = Math.max(fontSize + paddingY, point.y);

  context.fillStyle = fill;
  context.fillRect(x, y - fontSize - paddingY, metrics.width + paddingX * 2, fontSize + paddingY * 2);
  context.fillStyle = "#ffffff";
  context.fillText(text, x + paddingX, y);
}

function detectionLabel(detection: Detection, labelMode: OverlayLabelMode) {
  if (labelMode === "none") {
    return null;
  }
  if (labelMode === "patient") {
    return patientLabels[detection.class_name];
  }
  return `${detection.class_name} ${detection.confidence.toFixed(2)}`;
}

export default function OverlayCanvas({
  imageUrl,
  detections,
  confidenceFloor = 0.4,
  labelMode = "technical",
}: OverlayCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageUrl) {
      return;
    }

    const image = new Image();
    image.onload = () => {
      const width = image.naturalWidth || image.width;
      const height = image.naturalHeight || image.height;
      canvas.width = width;
      canvas.height = height;

      const context = canvas.getContext("2d");
      if (!context) {
        return;
      }

      context.clearRect(0, 0, width, height);
      context.drawImage(image, 0, 0, width, height);

      detections
        .filter((detection) => detection.confidence >= confidenceFloor)
        .forEach((detection) => {
          const colors = classColors[detection.class_name];
          const points = rotatedBox(detection.bbox);
          const lineWidth = detection.class_name === "post_acne_mark" ? 2 : 3;

          context.save();
          context.globalAlpha = colors.alpha;
          context.strokeStyle = colors.stroke;
          context.lineWidth = lineWidth;
          context.beginPath();
          points.forEach((point, index) => {
            if (index === 0) {
              context.moveTo(point.x, point.y);
            } else {
              context.lineTo(point.x, point.y);
            }
          });
          context.closePath();
          context.stroke();
          context.restore();

          const labelPoint = points.reduce((topLeft, point) => {
            if (point.y < topLeft.y || (point.y === topLeft.y && point.x < topLeft.x)) {
              return point;
            }
            return topLeft;
          }, points[0]);
          const label = detectionLabel(detection, labelMode);
          if (label) {
            drawLabel(
              context,
              label,
              labelPoint,
              colors.fill,
              detection.class_name === "post_acne_mark",
            );
          }
        });
    };
    image.src = imageUrl;
  }, [confidenceFloor, detections, imageUrl, labelMode]);

  return (
    <canvas
      ref={canvasRef}
      aria-label="Lesion detection overlay"
      className="block h-auto w-full rounded-md border border-stone-200 bg-stone-100"
    />
  );
}
