// Sanity-check the calculator's formulas against published training runs.
// Run with: node validate.js
// Same math as index.html: C = mult * N * D, GPU-hours = C / (peak * MFU * goodput) / 3600

const H100_BF16_PEAK = 989e12; // dense bf16, FLOP/s

const runs = [
  {
    name: "Llama 3.1 405B",
    N: 405e9,
    D: 15.6e12,
    gpus: 16384,
    reportedC: 3.8e25, // reported by Meta (Llama 3 paper)
    reportedGpuHours: 30.84e6, // Meta model card
    reportedDays: 54, // reported pre-training snapshot period, 16K H100s
  },
  {
    name: "Llama 3 70B (approximate N and D)",
    N: 70e9,
    D: 15e12,
    gpus: null,
    reportedC: null,
    reportedGpuHours: 6.4e6, // Meta model card
    reportedDays: null,
  },
];

function est({ N, D }, { mult = 6, mfu, goodput = 1, peak = H100_BF16_PEAK }) {
  const C = mult * N * D;
  const gpuHours = C / (peak * mfu * goodput) / 3600;
  return { C, gpuHours };
}

const fmt = (x) => x.toExponential(2);
const M = (x) => (x / 1e6).toFixed(1) + "M";
const pct = (a, b) => (((a - b) / b) * 100).toFixed(1) + "%";

for (const r of runs) {
  console.log("\n== " + r.name + " ==");
  const base = est(r, { mfu: 1, goodput: 1 });
  console.log("C = 6ND = " + fmt(base.C) + " FLOPs" + (r.reportedC ? "  (reported " + fmt(r.reportedC) + ", diff " + pct(base.C, r.reportedC) + ")" : ""));

  const scenarios = [
    { label: "MFU 41%, goodput 100%", mfu: 0.41, goodput: 1.0 },
    { label: "MFU 41%, goodput 90%", mfu: 0.41, goodput: 0.9 },
    { label: "MFU 38%, goodput 90%", mfu: 0.38, goodput: 0.9 },
  ];
  for (const s of scenarios) {
    const e = est(r, s);
    let line = s.label.padEnd(26) + " -> " + M(e.gpuHours) + " GPU-hours";
    if (r.reportedGpuHours) line += "  (vs reported " + M(r.reportedGpuHours) + ", " + pct(e.gpuHours, r.reportedGpuHours) + ")";
    if (r.gpus) line += "  ~" + (e.gpuHours / r.gpus / 24).toFixed(0) + " days on " + r.gpus + " GPUs";
    console.log(line);
  }

  if (r.reportedGpuHours) {
    const implied = base.C / (r.reportedGpuHours * 3600) / H100_BF16_PEAK;
    console.log("Implied overall utilization (MFU x goodput) from reported GPU-hours: " + (implied * 100).toFixed(1) + "%");
  }
}
