import { Marker } from "../components/Marker";
import { UShape } from "../components/UShape";
import { ASPPBlock, ASPP_BLOCK_H } from "../blocks/ASPPBlock";
import { P } from "../palette";

export function ASPPDiagram() {
  const { W, botY, els } = UShape({
    enc: [
      { t: "res", label: "enc1: ResidualEncoderBlock", ch: "3 → 64, s=1", badge: "R" },
      { t: "res", label: "enc2: ResidualEncoderBlock", ch: "64 → 128, s=2", badge: "R" },
      { t: "res", label: "enc3: ResidualEncoderBlock", ch: "128 → 256, s=2", badge: "R" },
      { t: "aspp", label: "enc4: ASPPBlock", ch: "256 → 512, s=2", badge: "A" },
    ],
    bot: { t: "aspp", label: "bottleneck: ASPPBlock", ch: "512 → 1024, s=2", badge: "A" },
    dec: [
      { t: "res", label: "dec1: ResidualDecoderBlock", ch: "1024 → 512", badge: "R" },
      { t: "res", label: "dec2: ResidualDecoderBlock", ch: "512 → 256", badge: "R" },
      { t: "res", label: "dec3: ResidualDecoderBlock", ch: "256 → 128", badge: "R" },
      { t: "res", label: "dec4: ResidualDecoderBlock", ch: "128 → 64", badge: "R" },
    ],
    skipStyle: (e) => ({ color: P[e.t].border }),
  });

  const blockY = botY + 20;
  const H = blockY + ASPP_BLOCK_H + 16;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ display: "block" }}>
      <defs>
        <Marker />
      </defs>
      {els}
      <ASPPBlock y={blockY} />
    </svg>
  );
}
