import { Marker } from "../components/Marker";
import { UShape } from "../components/UShape";
import { AGBlock, AG_BLOCK_H } from "../blocks/AGBlock";
import { P } from "../palette";

export function AGDiagram() {
  const agColor = P.ag.border;
  const { W, botY, els } = UShape({
    enc: [
      { t: "res", label: "enc1: ResidualEncoderBlock", ch: "3 → 64, s=1", badge: "R" },
      { t: "res", label: "enc2: ResidualEncoderBlock", ch: "64 → 128, s=2", badge: "R" },
      { t: "res", label: "enc3: ResidualEncoderBlock", ch: "128 → 256, s=2", badge: "R" },
      { t: "aspp", label: "enc4: ASPPBlock", ch: "256 → 512, s=2", badge: "A" },
    ],
    bot: { t: "aspp", label: "bottleneck: ASPPBlock", ch: "512 → 1024, s=2", badge: "A" },
    dec: [
      { t: "ag", label: "dec1: AG-ResDecoderBlock", ch: "1024 → 512", badge: "AG" },
      { t: "ag", label: "dec2: AG-ResDecoderBlock", ch: "512 → 256", badge: "AG" },
      { t: "ag", label: "dec3: AG-ResDecoderBlock", ch: "256 → 128", badge: "AG" },
      { t: "ag", label: "dec4: AG-ResDecoderBlock", ch: "128 → 64", badge: "AG" },
    ],
    skipStyle: () => ({ color: agColor }),
    skipMid: (midX, ey) => (
      <>
        <polygon
          points={`${midX},${ey - 13} ${midX + 14},${ey} ${midX},${ey + 13} ${midX - 14},${ey}`}
          fill={P.ag.bg}
          stroke={P.ag.border}
          strokeWidth={0.8}
        />
        <text
          x={midX}
          y={ey}
          textAnchor="middle"
          dominantBaseline="central"
          fontSize="7.5"
          fontWeight="700"
          fill={P.ag.text}
          fontFamily="system-ui"
        >
          AG
        </text>
      </>
    ),
  });

  const blockY = botY + 20;
  const H = blockY + AG_BLOCK_H + 16;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      <defs>
        <Marker />
      </defs>
      {els}
      <AGBlock y={blockY} />
    </svg>
  );
}
