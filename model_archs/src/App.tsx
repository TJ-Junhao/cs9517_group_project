import { useRef, useState } from "react";
import type { ComponentType } from "react";
import { ArchitectureDiagram } from "./diagrams/Architecture";
import { ModulesDiagram } from "./diagrams/Modules";
import { UNetDiagram } from "./diagrams/UNet";
import { ResUNetDiagram } from "./diagrams/ResUNet";
import { ASPPDiagram } from "./diagrams/ASPP";
import { AGDiagram } from "./diagrams/AG";
import { exportSvgAsPng } from "./utils/exportPng";
import { P } from "./palette";

type TabSpec = { id: string; label: string; C: ComponentType };

const TABS: TabSpec[] = [
  { id: "architecture", label: "Architecture", C: ArchitectureDiagram },
  { id: "modules", label: "Modules", C: ModulesDiagram },
  { id: "unet", label: "UNet", C: UNetDiagram },
  { id: "resunet", label: "ResUNet", C: ResUNetDiagram },
  { id: "aspp", label: "ASPP-ResUNet", C: ASPPDiagram },
  { id: "ag", label: "AG-ASPP-ResUNet", C: AGDiagram },
];

const LEGEND = [
  ["res", "R", "Residual"],
  ["aspp", "A", "ASPP"],
  ["ag", "AG", "Attn Gate"],
] as const;

export default function App() {
  const [tab, setTab] = useState(0);
  const [busy, setBusy] = useState(false);
  const canvasRef = useRef<HTMLDivElement>(null);
  const Comp = TABS[tab].C;

  const handleExport = async () => {
    const svg = canvasRef.current?.querySelector("svg");
    if (!svg) return;
    setBusy(true);
    try {
      await exportSvgAsPng(svg as SVGSVGElement, `${TABS[tab].id}.png`, 2);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ fontFamily: "system-ui, -apple-system, sans-serif", padding: "12px 0" }}>
      <div
        style={{
          display: "flex",
          gap: 6,
          marginBottom: 14,
          flexWrap: "wrap",
          alignItems: "center",
        }}
      >
        {TABS.map((t, i) => (
          <button
            key={t.id}
            onClick={() => setTab(i)}
            style={{
              padding: "7px 16px",
              borderRadius: 8,
              border: `1.2px solid ${tab === i ? "#6d63c7" : "#ddd"}`,
              background: tab === i ? "#eeedfd" : "transparent",
              color: tab === i ? "#2e2768" : "#888",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              transition: "all .15s",
            }}
          >
            {t.label}
          </button>
        ))}
        <button
          onClick={handleExport}
          disabled={busy}
          style={{
            marginLeft: "auto",
            padding: "7px 16px",
            borderRadius: 8,
            border: "1.2px solid #2a6aaa",
            background: busy ? "#eef2f7" : "#e8f0fa",
            color: "#163860",
            fontSize: 13,
            fontWeight: 600,
            cursor: busy ? "wait" : "pointer",
          }}
          title={`Export ${TABS[tab].id}.png`}
        >
          {busy ? "Exporting…" : `Export PNG (${TABS[tab].id})`}
        </button>
      </div>

      <div
        ref={canvasRef}
        style={{
          border: "1px solid #e8e6df",
          borderRadius: 12,
          padding: "12px 8px",
          background: "#fff",
        }}
      >
        <Comp />
      </div>

      <div
        style={{
          marginTop: 10,
          display: "flex",
          gap: 14,
          flexWrap: "wrap",
          fontSize: 11,
          color: "#999",
        }}
      >
        {LEGEND.map(([t, b, l]) => (
          <div key={t} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: 20,
                height: 18,
                borderRadius: 4,
                fontSize: 7.5,
                fontWeight: 700,
                background: P[t].tag,
                border: `1px solid ${P[t].border}`,
                color: P[t].sub,
              }}
            >
              {b}
            </span>
            {l}
          </div>
        ))}
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span style={{ display: "inline-block", width: 20, borderTop: "1.5px dashed #aaa" }} />
          Skip / Residual
        </div>
      </div>
    </div>
  );
}
