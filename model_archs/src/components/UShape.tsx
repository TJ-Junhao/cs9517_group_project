import { P } from "../palette";
import type { UShapeProps, UShapeResult } from "../types";

export function UShape({
  enc,
  dec,
  bot,
  skipStyle,
  skipMid,
  outLabel = "Conv 1×1 → Output",
}: UShapeProps): UShapeResult {
  const W = 700;
  const EX = 30;
  const DX = 440;
  const BW = 220;
  const bh = 56;
  const gap = 14;
  const TOP = 60;

  const encPos = enc.map((_, i) => ({ x: EX, y: TOP + i * (bh + gap), w: BW, h: bh }));
  const botYTop = TOP + enc.length * (bh + gap) + 14;
  const botW = 250;
  const botX = W / 2 - botW / 2;
  const decPos = dec.map((_, i) => ({
    x: DX,
    y: encPos[enc.length - 1 - i].y,
    w: BW,
    h: bh,
  }));
  const outY = encPos[0].y - 38;

  const encCx = EX + BW / 2;
  const decCx = DX + BW / 2;
  const botCx = W / 2;
  const bendGap = 10;

  return {
    W,
    botY: botYTop + bh + 30,
    els: (
      <>
        <text
          x={encCx}
          y={12}
          textAnchor="middle"
          fontSize="11"
          fill="#aaa"
          fontFamily="system-ui"
          fontWeight="500"
        >
          ENCODER
        </text>
        <text
          x={decCx}
          y={12}
          textAnchor="middle"
          fontSize="11"
          fill="#aaa"
          fontFamily="system-ui"
          fontWeight="500"
        >
          DECODER
        </text>

        {/* encoder blocks */}
        {enc.map((e, i) => {
          const c = P[e.t];
          const ep = encPos[i];
          return (
            <g key={`e${i}`}>
              <rect
                x={ep.x}
                y={ep.y}
                width={ep.w}
                height={ep.h}
                rx={8}
                fill={c.bg}
                stroke={c.border}
                strokeWidth={0.8}
              />
              <text
                x={ep.x + 12}
                y={ep.y + 22}
                fontSize="11.5"
                fontWeight="600"
                fill={c.text}
                fontFamily="system-ui"
              >
                {e.label}
              </text>
              <text x={ep.x + 12} y={ep.y + 40} fontSize="10" fill={c.sub} fontFamily="system-ui">
                {e.ch}
              </text>
              {e.badge && (
                <>
                  <rect
                    x={ep.x + ep.w - 24}
                    y={ep.y + 4}
                    width={20}
                    height={16}
                    rx={3}
                    fill={c.tag}
                    stroke={c.border}
                    strokeWidth={0.4}
                  />
                  <text
                    x={ep.x + ep.w - 14}
                    y={ep.y + 12}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize="7.5"
                    fontWeight="700"
                    fill={c.sub}
                    fontFamily="system-ui"
                  >
                    {e.badge}
                  </text>
                </>
              )}
              {i < enc.length - 1 && (
                <line
                  x1={encCx}
                  y1={ep.y + bh}
                  x2={encCx}
                  y2={encPos[i + 1].y}
                  stroke="#bbb"
                  strokeWidth={1.2}
                  markerEnd="url(#ah)"
                />
              )}
            </g>
          );
        })}

        {/* enc last → bottleneck (L-bend) */}
        <path
          d={`M${encCx} ${encPos[enc.length - 1].y + bh} L${encCx} ${botYTop - bendGap} L${botCx} ${botYTop - bendGap} L${botCx} ${botYTop}`}
          fill="none"
          stroke="#bbb"
          strokeWidth={1.2}
          markerEnd="url(#ah)"
        />

        {/* bottleneck */}
        <rect
          x={botX}
          y={botYTop}
          width={botW}
          height={bh}
          rx={8}
          fill={P[bot.t].bg}
          stroke={P[bot.t].border}
          strokeWidth={0.8}
        />
        <text
          x={botCx}
          y={botYTop + 22}
          textAnchor="middle"
          fontSize="12"
          fontWeight="600"
          fill={P[bot.t].text}
          fontFamily="system-ui"
        >
          {bot.label}
        </text>
        <text
          x={botCx}
          y={botYTop + 40}
          textAnchor="middle"
          fontSize="10"
          fill={P[bot.t].sub}
          fontFamily="system-ui"
        >
          {bot.ch}
        </text>
        {bot.badge && (
          <>
            <rect
              x={botX + botW - 24}
              y={botYTop + 4}
              width={20}
              height={16}
              rx={3}
              fill={P[bot.t].tag}
              stroke={P[bot.t].border}
              strokeWidth={0.4}
            />
            <text
              x={botX + botW - 14}
              y={botYTop + 12}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="7.5"
              fontWeight="700"
              fill={P[bot.t].sub}
              fontFamily="system-ui"
            >
              {bot.badge}
            </text>
          </>
        )}

        {/* bottleneck → dec first (L-bend) */}
        <path
          d={`M${botCx} ${botYTop + bh} L${botCx} ${botYTop + bh + bendGap} L${decCx} ${botYTop + bh + bendGap} L${decCx} ${decPos[0].y + bh}`}
          fill="none"
          stroke="#bbb"
          strokeWidth={1.2}
          markerEnd="url(#ah)"
        />

        {/* decoder blocks */}
        {dec.map((d, i) => {
          const c = P[d.t];
          const dp = decPos[i];
          return (
            <g key={`d${i}`}>
              <rect
                x={dp.x}
                y={dp.y}
                width={dp.w}
                height={dp.h}
                rx={8}
                fill={c.bg}
                stroke={c.border}
                strokeWidth={0.8}
              />
              <text
                x={dp.x + 12}
                y={dp.y + 22}
                fontSize="11.5"
                fontWeight="600"
                fill={c.text}
                fontFamily="system-ui"
              >
                {d.label}
              </text>
              <text x={dp.x + 12} y={dp.y + 40} fontSize="10" fill={c.sub} fontFamily="system-ui">
                {d.ch}
              </text>
              {d.badge && (
                <>
                  <rect
                    x={dp.x + dp.w - 28}
                    y={dp.y + 4}
                    width={24}
                    height={16}
                    rx={3}
                    fill={c.tag}
                    stroke={c.border}
                    strokeWidth={0.4}
                  />
                  <text
                    x={dp.x + dp.w - 16}
                    y={dp.y + 12}
                    textAnchor="middle"
                    dominantBaseline="central"
                    fontSize="7"
                    fontWeight="700"
                    fill={c.sub}
                    fontFamily="system-ui"
                  >
                    {d.badge}
                  </text>
                </>
              )}
              {i < dec.length - 1 && (
                <line
                  x1={decCx}
                  y1={dp.y}
                  x2={decCx}
                  y2={decPos[i + 1].y + bh}
                  stroke="#bbb"
                  strokeWidth={1.2}
                  markerEnd="url(#ah)"
                />
              )}
            </g>
          );
        })}

        {/* skip connections */}
        {enc.map((e, i) => {
          const ey = encPos[i].y + bh / 2;
          const sc = skipStyle ? skipStyle(e, i) : { color: P[e.t].border };
          const midX = (EX + BW + DX) / 2;
          return (
            <g key={`s${i}`}>
              {skipMid ? (
                <>
                  <line
                    x1={EX + BW + 6}
                    y1={ey}
                    x2={midX - 16}
                    y2={ey}
                    stroke={sc.color}
                    strokeWidth={1}
                    strokeDasharray="6 3"
                    markerEnd="url(#ah)"
                  />
                  {skipMid(midX, ey, i)}
                  <line
                    x1={midX + 16}
                    y1={ey}
                    x2={DX - 6}
                    y2={ey}
                    stroke={sc.color}
                    strokeWidth={1}
                    markerEnd="url(#ah)"
                  />
                </>
              ) : (
                <>
                  <line
                    x1={EX + BW + 6}
                    y1={ey}
                    x2={DX - 6}
                    y2={ey}
                    stroke={sc.color}
                    strokeWidth={1}
                    strokeDasharray="6 3"
                    markerEnd="url(#ah)"
                  />
                  <text
                    x={midX}
                    y={ey - 6}
                    textAnchor="middle"
                    fontSize="9"
                    fill="#bbb"
                    fontFamily="system-ui"
                  >
                    skip
                  </text>
                </>
              )}
            </g>
          );
        })}

        {/* output */}
        <rect
          x={DX + 40}
          y={outY}
          width={140}
          height={32}
          rx={7}
          fill={P.out.bg}
          stroke={P.out.border}
          strokeWidth={0.8}
        />
        <text
          x={DX + 110}
          y={outY + 16}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize="11"
          fontWeight="600"
          fill={P.out.text}
          fontFamily="system-ui"
        >
          {outLabel}
        </text>
        <line
          x1={DX + 110}
          y1={outY + 32}
          x2={DX + 110}
          y2={decPos[dec.length - 1].y}
          stroke="#bbb"
          strokeWidth={1.2}
          markerEnd="url(#ah)"
        />
      </>
    ),
  };
}
