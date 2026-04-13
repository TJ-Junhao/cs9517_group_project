export type Swatch = {
  bg: string;
  border: string;
  text: string;
  sub: string;
  tag: string;
};

export const P = {
  conv: { bg: "#f5f4f0", border: "#a8a69e", text: "#3d3b36", sub: "#7a786f", tag: "#e8e6df" },
  res: { bg: "#eeedfd", border: "#8b83e0", text: "#2e2768", sub: "#6d63c7", tag: "#dddaf9" },
  aspp: { bg: "#e4f6f0", border: "#2daa7f", text: "#0a5040", sub: "#1a8060", tag: "#c5eddd" },
  ag: { bg: "#fdeee8", border: "#e06838", text: "#5a2010", sub: "#b84820", tag: "#fad4c4" },
  out: { bg: "#e8f0fa", border: "#4a90d0", text: "#163860", sub: "#2a6aaa", tag: "#c8ddf5" },
  bot: { bg: "#fdf4e2", border: "#c89030", text: "#4a3510", sub: "#9a7020", tag: "#f5e4b8" },
} satisfies Record<string, Swatch>;

export type Kind = keyof typeof P;
