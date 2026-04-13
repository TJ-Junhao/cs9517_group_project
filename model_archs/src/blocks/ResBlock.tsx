import { P } from "../palette";

export const RES_BLOCK_H = 140;

export function ResBlock({ y }: { y: number }) {
  const c = P.res;
  const by = y + 38;
  const nodes = [
    { x: 40, w: 55, l: "Input", plain: true },
    { x: 118, w: 72, l: "Conv 3×3" },
    { x: 212, w: 82, l: "GN → ReLU" },
    { x: 316, w: 72, l: "Conv 3×3" },
    { x: 410, w: 72, l: "GN" },
  ];
  const addCx = 510;
  const reluX = 548;

  return (
    <g>
      <rect
        x={25}
        y={y}
        width={650}
        height={RES_BLOCK_H}
        rx={10}
        fill="#fafaf8"
        stroke="#e0ddd5"
        strokeWidth={0.6}
      />
      <text x={45} y={y + 18} fontSize="11.5" fontWeight="700" fill={c.text} fontFamily="system-ui">
        ResidualEncoderBlock internals
      </text>
      {nodes.map((n, i) => (
        <g key={i}>
          <rect
            x={n.x}
            y={by}
            width={n.w}
            height={32}
            rx={6}
            fill={n.plain ? "#f0efeb" : c.tag}
            stroke={n.plain ? "#ccc" : c.border}
            strokeWidth={0.5}
          />
          <text
            x={n.x + n.w / 2}
            y={by + 16}
            textAnchor="middle"
            dominantBaseline="central"
            fontSize="10"
            fontWeight="500"
            fill={c.text}
            fontFamily="system-ui"
          >
            {n.l}
          </text>
          {i < nodes.length - 1 && (
            <line
              x1={n.x + n.w + 2}
              y1={by + 16}
              x2={nodes[i + 1].x - 2}
              y2={by + 16}
              stroke="#bbb"
              strokeWidth={0.8}
              markerEnd="url(#ah)"
            />
          )}
        </g>
      ))}
      <line
        x1={484}
        y1={by + 16}
        x2={addCx - 14}
        y2={by + 16}
        stroke="#bbb"
        strokeWidth={0.8}
        markerEnd="url(#ah)"
      />
      <circle cx={addCx} cy={by + 16} r={14} fill={c.tag} stroke={c.border} strokeWidth={0.8} />
      <text
        x={addCx}
        y={by + 16}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="16"
        fontWeight="700"
        fill={c.text}
      >
        +
      </text>
      <line
        x1={addCx + 14}
        y1={by + 16}
        x2={reluX + 2}
        y2={by + 16}
        stroke="#bbb"
        strokeWidth={0.8}
        markerEnd="url(#ah)"
      />
      <rect
        x={reluX + 4}
        y={by}
        width={52}
        height={32}
        rx={6}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.5}
      />
      <text
        x={reluX + 30}
        y={by + 16}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="10"
        fontWeight="500"
        fill={c.text}
        fontFamily="system-ui"
      >
        ReLU
      </text>
      <path
        d={`M67 ${by + 32} L67 ${by + 72} L${addCx} ${by + 72} L${addCx} ${by + 30}`}
        fill="none"
        stroke={c.border}
        strokeWidth={1.2}
        strokeDasharray="5 3"
        markerEnd="url(#ah)"
      />
      <rect
        x={215}
        y={by + 60}
        width={130}
        height={24}
        rx={5}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.5}
      />
      <text
        x={280}
        y={by + 72}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="10"
        fontWeight="600"
        fill={c.text}
        fontFamily="system-ui"
      >
        Identity / 1×1 proj
      </text>
      <text
        x={280}
        y={by + 96}
        textAnchor="middle"
        fontSize="9"
        fill={c.sub}
        fontFamily="system-ui"
        fontStyle="italic"
      >
        Residual: 1×1 projection when channel_in ≠ channel_out
      </text>
    </g>
  );
}
