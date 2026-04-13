import { P } from "../palette";

export const ASPP_BLOCK_H = 228;

export function ASPPBlock({ y }: { y: number }) {
  const c = P.aspp;
  const by = y + 38;
  const branchLabels = ["1×1 conv", "3×3, d=3", "3×3, d=6", "3×3, d=9"];
  const branchGap = 42;
  const branchH = 30;
  const branchStartY = by + 4;
  const inputCY = branchStartY + (3 * branchGap + branchH) / 2;
  const idY = branchStartY + 3 * branchGap + branchH + 18;

  return (
    <g>
      <rect
        x={25}
        y={y}
        width={650}
        height={ASPP_BLOCK_H}
        rx={10}
        fill="#fafaf8"
        stroke="#e0ddd5"
        strokeWidth={0.6}
      />
      <text x={45} y={y + 18} fontSize="11.5" fontWeight="700" fill={c.text} fontFamily="system-ui">
        ASPPBlock internals (Atrous Spatial Pyramid Pooling)
      </text>

      {/* Input */}
      <rect
        x={35}
        y={inputCY - 15}
        width={60}
        height={30}
        rx={5}
        fill="#f0efeb"
        stroke="#ccc"
        strokeWidth={0.5}
      />
      <text
        x={65}
        y={inputCY}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="10"
        fill="#555"
        fontFamily="system-ui"
      >
        Input
      </text>

      {/* Initial Conv */}
      <line
        x1={95}
        y1={inputCY}
        x2={113}
        y2={inputCY}
        stroke="#bbb"
        strokeWidth={0.8}
        markerEnd="url(#ah)"
      />
      <rect
        x={115}
        y={inputCY - 18}
        width={65}
        height={36}
        rx={5}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.5}
      />
      <text
        x={147}
        y={inputCY - 4}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="9.5"
        fontWeight="500"
        fill={c.text}
        fontFamily="system-ui"
      >
        Conv 3×3
      </text>
      <text
        x={147}
        y={inputCY + 10}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="8"
        fill={c.sub}
        fontFamily="system-ui"
      >
        GN+ReLU
      </text>

      {/* 4 branches */}
      {branchLabels.map((bl, i) => {
        const bcy = branchStartY + i * branchGap + branchH / 2;
        return (
          <g key={i}>
            <path
              d={`M180 ${inputCY} L203 ${bcy}`}
              fill="none"
              stroke={c.border}
              strokeWidth={0.7}
              markerEnd="url(#ah)"
            />
            <rect
              x={205}
              y={bcy - branchH / 2}
              width={80}
              height={branchH}
              rx={5}
              fill={c.tag}
              stroke={c.border}
              strokeWidth={0.5}
            />
            <text
              x={245}
              y={bcy - 4}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="9.5"
              fontWeight="500"
              fill={c.text}
              fontFamily="system-ui"
            >
              {bl}
            </text>
            <text
              x={245}
              y={bcy + 9}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="7.5"
              fill={c.sub}
              fontFamily="system-ui"
            >
              GN + ReLU
            </text>
            <path
              d={`M285 ${bcy} L313 ${inputCY}`}
              fill="none"
              stroke={c.border}
              strokeWidth={0.7}
              markerEnd="url(#ah)"
            />
          </g>
        );
      })}

      {/* Concat */}
      <rect
        x={315}
        y={inputCY - 18}
        width={48}
        height={36}
        rx={5}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.7}
      />
      <text
        x={339}
        y={inputCY - 4}
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
        x={339}
        y={inputCY + 10}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="8"
        fill={c.sub}
        fontFamily="system-ui"
      >
        C×4
      </text>

      {/* 1×1 proj + GN */}
      <line
        x1={363}
        y1={inputCY}
        x2={383}
        y2={inputCY}
        stroke="#bbb"
        strokeWidth={0.8}
        markerEnd="url(#ah)"
      />
      <rect
        x={385}
        y={inputCY - 18}
        width={60}
        height={36}
        rx={5}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.5}
      />
      <text
        x={415}
        y={inputCY - 4}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="9.5"
        fontWeight="500"
        fill={c.text}
        fontFamily="system-ui"
      >
        1×1 proj
      </text>
      <text
        x={415}
        y={inputCY + 10}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="8"
        fill={c.sub}
        fontFamily="system-ui"
      >
        GN
      </text>

      {/* Add */}
      <line
        x1={445}
        y1={inputCY}
        x2={465}
        y2={inputCY}
        stroke="#bbb"
        strokeWidth={0.8}
        markerEnd="url(#ah)"
      />
      <circle cx={480} cy={inputCY} r={14} fill={c.tag} stroke={c.border} strokeWidth={0.8} />
      <text
        x={480}
        y={inputCY}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="16"
        fontWeight="700"
        fill={c.text}
      >
        +
      </text>

      {/* ReLU */}
      <line
        x1={494}
        y1={inputCY}
        x2={518}
        y2={inputCY}
        stroke="#bbb"
        strokeWidth={0.8}
        markerEnd="url(#ah)"
      />
      <rect
        x={520}
        y={inputCY - 15}
        width={50}
        height={30}
        rx={5}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.5}
      />
      <text
        x={545}
        y={inputCY}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize="10"
        fontWeight="500"
        fill={c.text}
        fontFamily="system-ui"
      >
        ReLU
      </text>

      {/* Identity residual */}
      <path
        d={`M65 ${inputCY + 15} L65 ${idY} L480 ${idY} L480 ${inputCY + 14}`}
        fill="none"
        stroke={c.border}
        strokeWidth={1.2}
        strokeDasharray="5 3"
        markerEnd="url(#ah)"
      />
      <rect
        x={210}
        y={idY - 12}
        width={130}
        height={24}
        rx={5}
        fill={c.tag}
        stroke={c.border}
        strokeWidth={0.5}
      />
      <text
        x={275}
        y={idY}
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
        x={340}
        y={idY + 22}
        textAnchor="middle"
        fontSize="9"
        fill={c.sub}
        fontFamily="system-ui"
        fontStyle="italic"
      >
        4 parallel atrous convolutions (d=1,3,6,9) + residual connection
      </text>
    </g>
  );
}
