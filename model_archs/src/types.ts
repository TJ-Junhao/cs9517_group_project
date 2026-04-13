import type { ReactNode } from "react";
import type { Kind } from "./palette";

export type Block = {
  t: Kind;
  label: string;
  ch: string;
  badge?: string;
};

export type Skip = { color: string };

export type UShapeProps = {
  enc: Block[];
  dec: Block[];
  bot: Block;
  skipStyle?: (e: Block, i: number) => Skip;
  skipMid?: (midX: number, ey: number, i: number) => ReactNode;
  outLabel?: string;
};

export type UShapeResult = {
  W: number;
  botY: number;
  els: ReactNode;
};
