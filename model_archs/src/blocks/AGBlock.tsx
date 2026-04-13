import { P } from "../palette";

export const AG_BLOCK_H = 248;

export function AGBlock({ y }: { y: number }) {
  const c = P.ag;
  const rc = P.res;
  const by = y + 42;

  // Landmarks (all relative to by)
  const skipRight = 113;
  const decoderRight = 113;
  const convtLeft = 137;
  const convtRight = 205;
  const wscLeft = 142;
  const wscRight = 190;
  const wupLeft = 204;
  const wupRight = 252;
  const sumCx = 278;
  const sumCy = 32;
  const sumR = 10;
  const reluLeft = 305;
  const reluRight = 365;
  const mulCx = 398;
  const mulCy = 32;
  const mulR = 10;
  const catLeft = 418;
  const catRight = 462;
  const catTopY = 88;
  const resLeft = 483;

  // Branch points
  const branchX = 215; // where ConvT output splits to W_up and Cat
  const catUpsampleX = 432; // x where upsample enters Cat top
  const catGatedX = 448; // x where gated skip enters Cat top

  return (
    <g>
      <rect
        x={25}
        y={y}
        width={650}
        height={AG_BLOCK_H}
        rx={10}
        fill="#fafaf8"
        stroke="#e0ddd5"
        strokeWidth={0.6}
      />
      <text x={45} y={y + 18} fontSize="11.5" fontWeight="700" fill={c.text} fontFamily="system-ui">
        AttentionGate + ResidualDecoder internals
      </text>

      {/* Inputs */}
      <rect
        x={35}
        y={by}
        width={78}
        height={28}
        rx={5}
        fill="#f0efeb"
        stroke="#ccc"
        strokeWidth={0.5}
      />
      <text
        x={74}
        y={by + 14}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="10"
        fill="#555"
        fontFamily="system-ui"
      >
        Skip (x₂)
      </text>
      <rect
        x={35}
        y={by + 88}
        width={78}
        height={28}
        rx={5}
        fill="#f0efeb"
        stroke="#ccc"
        strokeWidth={0.5}
      />
      <text
        x={74}
        y={by + 102}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="10"
        fill="#555"
        fontFamily="system-ui"
      >
        Decoder (x₁)
      </text>

      {/* Decoder → ConvT */}
      <line
        x1={decoderRight}
        y1={by + 102}
        x2={convtLeft - 2}
        y2={by + 102}
        stroke="#bbb"
        strokeWidth={0.8}
        markerEnd="url(#ah)"
      />
      <rect
        x={convtLeft}
        y={by + 88}
        width={68}
        height={28}
        rx={5}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.5}
      />
      <text
        x={171}
        y={by + 102}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="9.5"
        fontWeight="500"
        fill={c.text}
        fontFamily="system-ui"
      >
        ConvT ↑2
      </text>

      {/* AG dashed enclosure */}
      <rect
        x={120}
        y={by - 16}
        width={260}
        height={72}
        rx={7}
        fill="none"
        stroke={c.border}
        strokeWidth={0.6}
        strokeDasharray="4 2"
      />
      <text
        x={250}
        y={by - 4}
        textAnchor="middle"
        fontSize="9"
        fontWeight="600"
        fill={c.sub}
        fontFamily="system-ui"
      >
        Attention Gate
      </text>

      {/* Skip → W_sc */}
      <line
        x1={skipRight}
        y1={by + 14}
        x2={wscLeft - 2}
        y2={by + 14}
        stroke="#bbb"
        strokeWidth={0.7}
        markerEnd="url(#ah)"
      />
      <rect
        x={wscLeft}
        y={by + 2}
        width={48}
        height={24}
        rx={4}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.4}
      />
      <text
        x={166}
        y={by + 14}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="8.5"
        fill={c.text}
        fontFamily="system-ui"
      >
        W_sc
      </text>

      {/* W_up box */}
      <rect
        x={wupLeft}
        y={by + 2}
        width={48}
        height={24}
        rx={4}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.4}
      />
      <text
        x={228}
        y={by + 14}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="8.5"
        fill={c.text}
        fontFamily="system-ui"
      >
        W_up
      </text>

      {/* ConvT output → branch */}
      <line
        x1={convtRight}
        y1={by + 102}
        x2={branchX}
        y2={by + 102}
        stroke="#bbb"
        strokeWidth={0.7}
      />
      <circle cx={branchX} cy={by + 102} r={2} fill="#888" />

      {/* branch → W_up bottom */}
      <line
        x1={branchX}
        y1={by + 102}
        x2={branchX}
        y2={by + 28}
        stroke="#bbb"
        strokeWidth={0.7}
        markerEnd="url(#ah)"
      />

      {/* branch → Cat top (upsampled feature) */}
      <path
        d={`M${branchX} ${by + 102} L${catUpsampleX} ${by + 102} L${catUpsampleX} ${by + catTopY + 2}`}
        fill="none"
        stroke="#bbb"
        strokeWidth={0.7}
        markerEnd="url(#ah)"
      />

      {/* W_sc output → sum (go down then right, avoiding W_up) */}
      <path
        d={`M${wscRight} ${by + 14} L${wscRight + 6} ${by + 14} L${wscRight + 6} ${by + sumCy} L${sumCx - sumR - 1} ${by + sumCy}`}
        fill="none"
        stroke="#bbb"
        strokeWidth={0.7}
        markerEnd="url(#ah)"
      />

      {/* W_up output → sum top (right then down) */}
      <path
        d={`M${wupRight} ${by + 14} L${sumCx} ${by + 14} L${sumCx} ${by + sumCy - sumR - 1}`}
        fill="none"
        stroke="#bbb"
        strokeWidth={0.7}
        markerEnd="url(#ah)"
      />

      {/* Sum + */}
      <circle
        cx={sumCx}
        cy={by + sumCy}
        r={sumR}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.5}
      />
      <text
        x={sumCx}
        y={by + sumCy}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="12"
        fontWeight="700"
        fill={c.text}
      >
        +
      </text>

      {/* Sum → ReLU/Conv/Sigmoid */}
      <line
        x1={sumCx + sumR}
        y1={by + sumCy}
        x2={reluLeft - 2}
        y2={by + sumCy}
        stroke="#bbb"
        strokeWidth={0.7}
        markerEnd="url(#ah)"
      />
      <rect
        x={reluLeft}
        y={by + 10}
        width={60}
        height={44}
        rx={4}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.4}
      />
      <text
        x={335}
        y={by + 22}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="8.5"
        fill={c.text}
        fontFamily="system-ui"
      >
        ReLU
      </text>
      <text
        x={335}
        y={by + 34}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="8.5"
        fill={c.text}
        fontFamily="system-ui"
      >
        Conv 1×1
      </text>
      <text
        x={335}
        y={by + 47}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="9"
        fill={c.text}
        fontFamily="system-ui"
      >
        Sigmoid
      </text>

      {/* Sigmoid α → Multiply */}
      <line
        x1={reluRight}
        y1={by + mulCy}
        x2={mulCx - mulR - 1}
        y2={by + mulCy}
        stroke="#bbb"
        strokeWidth={0.7}
        markerEnd="url(#ah)"
      />

      {/* Multiply circle */}
      <circle
        cx={mulCx}
        cy={by + mulCy}
        r={mulR}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.5}
      />
      <text
        x={mulCx}
        y={by + mulCy}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="12"
        fontWeight="700"
        fill={c.text}
      >
        ×
      </text>
      <text
        x={mulCx}
        y={by + 17}
        textAnchor="middle"
        fontSize="7.5"
        fill={c.sub}
        fontFamily="system-ui"
      >
        gated
      </text>

      {/* Skip x₂ bypass → Multiply bottom */}
      <path
        d={`M74 ${by + 28} L74 ${by + 66} L${mulCx} ${by + 66} L${mulCx} ${by + mulCy + mulR + 1}`}
        fill="none"
        stroke={c.border}
        strokeWidth={0.8}
        markerEnd="url(#ah)"
      />

      {/* Multiply → Cat top (gated skip) */}
      <path
        d={`M${mulCx + mulR} ${by + mulCy} L${catGatedX} ${by + mulCy} L${catGatedX} ${by + catTopY + 2}`}
        fill="none"
        stroke="#bbb"
        strokeWidth={0.7}
        markerEnd="url(#ah)"
      />

      {/* Cat */}
      <rect
        x={catLeft}
        y={by + catTopY}
        width={44}
        height={28}
        rx={5}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.6}
      />
      <text
        x={440}
        y={by + catTopY + 10}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="10"
        fontWeight="600"
        fill={c.text}
        fontFamily="system-ui"
      >
        Cat
      </text>
      <text
        x={440}
        y={by + catTopY + 22}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="7.5"
        fill={c.sub}
        fontFamily="system-ui"
      >
        C×2
      </text>

      {/* Cat → Residual conv block */}
      <line
        x1={catRight}
        y1={by + 102}
        x2={resLeft - 2}
        y2={by + 102}
        stroke="#bbb"
        strokeWidth={0.7}
        markerEnd="url(#ah)"
      />

      {/* Residual conv block */}
      <rect
        x={resLeft}
        y={by + 78}
        width={140}
        height={72}
        rx={6}
        fill={rc.bg}
        stroke={rc.border}
        strokeWidth={0.6}
      />
      <text
        x={553}
        y={by + 94}
        textAnchor="middle"
        fontSize="9.5"
        fontWeight="600"
        fill={rc.text}
        fontFamily="system-ui"
      >
        Conv 3×3 → GN → ReLU
      </text>
      <text
        x={553}
        y={by + 110}
        textAnchor="middle"
        fontSize="9.5"
        fontWeight="600"
        fill={rc.text}
        fontFamily="system-ui"
      >
        Conv 3×3 → GN
      </text>
      <text
        x={553}
        y={by + 130}
        textAnchor="middle"
        fontSize="9"
        fill={rc.sub}
        fontFamily="system-ui"
      >
        + identity → ReLU
      </text>

      {/* Residual bypass inside decoder block */}
      <path
        d={`M${resLeft} ${by + 102} L${resLeft - 4} ${by + 102} L${resLeft - 4} ${by + 170} L553 ${by + 170} L553 ${by + 150}`}
        fill="none"
        stroke={rc.border}
        strokeWidth={1}
        strokeDasharray="5 3"
        markerEnd="url(#ah)"
      />
      <text x={490} y={by + 182} fontSize="8" fill={rc.sub} fontFamily="system-ui">
        residual path
      </text>

      {/* Summary */}
      <text x={45} y={by + 203} fontSize="9" fill={c.sub} fontFamily="system-ui" fontStyle="italic">
        Attention gate: sigmoid produces spatial weight map α∈[0,1] to suppress irrelevant skip
        features
      </text>
    </g>
  );
}
