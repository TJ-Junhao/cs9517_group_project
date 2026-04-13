import { Marker } from "../components/Marker";
import { ResBlock, RES_BLOCK_H } from "../blocks/ResBlock";
import { ASPPBlock, ASPP_BLOCK_H } from "../blocks/ASPPBlock";
import { AGBlock, AG_BLOCK_H } from "../blocks/AGBlock";

export function ModulesDiagram() {
  const W = 700;
  const PAD_TOP = 24;
  const GAP = 22;

  const resY = PAD_TOP;
  const asppY = resY + RES_BLOCK_H + GAP;
  const agY = asppY + ASPP_BLOCK_H + GAP;
  const H = agY + AG_BLOCK_H + 16;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      <defs>
        <Marker />
      </defs>
      <text
        x={W / 2}
        y={14}
        textAnchor="middle"
        fontSize="12"
        fontWeight="700"
        fill="#555"
        fontFamily="system-ui"
      >
        AG-ASPP-ResU-Net — module internals
      </text>
      <ResBlock y={resY} />
      <ASPPBlock y={asppY} />
      <AGBlock y={agY} />
    </svg>
  );
}
