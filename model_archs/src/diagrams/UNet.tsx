import { Marker } from "../components/Marker";
import { UShape } from "../components/UShape";
import { P } from "../palette";

export function UNetDiagram() {
  const { W, botY, els } = UShape({
    enc: [
      { t: "conv", label: "enc1: ConvBlock", ch: "3 → 64" },
      { t: "conv", label: "enc2: MaxPool → ConvBlock", ch: "64 → 128" },
      { t: "conv", label: "enc3: MaxPool → ConvBlock", ch: "128 → 256" },
      { t: "conv", label: "enc4: MaxPool → ConvBlock", ch: "256 → 512" },
    ],
    bot: { t: "bot", label: "enc5 (bottleneck): MaxPool → ConvBlock", ch: "512 → 1024" },
    dec: [
      { t: "conv", label: "dec1: ConvT↑ → Cat → ConvBlock", ch: "1024 → 512" },
      { t: "conv", label: "dec2: ConvT↑ → Cat → ConvBlock", ch: "512 → 256" },
      { t: "conv", label: "dec3: ConvT↑ → Cat → ConvBlock", ch: "256 → 128" },
      { t: "conv", label: "dec4: ConvT↑ → Cat → ConvBlock", ch: "128 → 64" },
    ],
  });
  const dY = botY + 20;
  const H = dY + 80;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      <defs>
        <Marker />
      </defs>
      {els}
      <rect
        x={25}
        y={dY}
        width={650}
        height={70}
        rx={10}
        fill="#fafaf8"
        stroke="#e0ddd5"
        strokeWidth={0.6}
      />
      <text x={45} y={dY + 18} fontSize="11" fontWeight="600" fill="#555" fontFamily="system-ui">
        ConvBlock internals
      </text>
      {["Input", "Conv 3×3", "ReLU", "Conv 3×3", "ReLU", "Output"].map((l, i) => {
        const bx = 45 + i * 105;
        const by = dY + 28;
        const isEdge = i === 0 || i === 5;
        return (
          <g key={i}>
            <rect
              x={bx}
              y={by}
              width={85}
              height={28}
              rx={5}
              fill={isEdge ? "#f0efeb" : P.conv.tag}
              stroke={isEdge ? "#ccc" : P.conv.border}
              strokeWidth={0.5}
            />
            <text
              x={bx + 42}
              y={by + 14}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="10"
              fontWeight="500"
              fill={P.conv.text}
              fontFamily="system-ui"
            >
              {l}
            </text>
            {i < 5 && (
              <line
                x1={bx + 85 + 2}
                y1={by + 14}
                x2={bx + 105 - 2}
                y2={by + 14}
                stroke="#bbb"
                strokeWidth={0.8}
                markerEnd="url(#ah)"
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
