import React from "react";

// Single 1px line, no fill, no gradient, no glow, per the design doc.

export function Sparkline({ values, width = 220, height = 36 }) {
  if (!values || values.length < 2) return <div className="empty" style={{ padding: 8 }}>No trades placed yet</div>;
  const min = Math.min(...values), max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) =>
    `${(i / (values.length - 1)) * width},${height - ((v - min) / span) * height}`).join(" ");
  return (
    <svg width={width} height={height} style={{ display: "block", width: "100%" }}
      viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
      <polyline points={pts} fill="none" stroke="#ECEEF0" strokeWidth="1" />
    </svg>
  );
}

export function EquityCurve({ points, target, floor, height = 260 }) {
  if (!points || points.length < 2)
    return <div className="empty">No trades placed yet</div>;
  const w = 1000;
  const values = points.map((p) => p.equity);
  const min = Math.min(...values, floor ?? Infinity);
  const max = Math.max(...values, target ?? -Infinity);
  const span = max - min || 1;
  const y = (v) => height - ((v - min) / span) * (height - 20) - 10;
  const pts = values.map((v, i) => `${(i / (values.length - 1)) * w},${y(v)}`).join(" ");
  return (
    <svg width="100%" height={height} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none">
      <line x1="0" y1={height - 1} x2={w} y2={height - 1} stroke="#272C33" strokeWidth="1" />
      {target != null && (
        <g>
          <line x1="0" y1={y(target)} x2={w} y2={y(target)} stroke="#8A93A1" strokeWidth="1" strokeDasharray="6 4" />
          <text x="8" y={y(target) - 4} fill="#8A93A1" fontSize="11" fontFamily="IBM Plex Mono">PROFIT TARGET</text>
        </g>
      )}
      {floor != null && (
        <g>
          <line x1="0" y1={y(floor)} x2={w} y2={y(floor)} stroke="#8A93A1" strokeWidth="1" strokeDasharray="6 4" />
          <text x="8" y={y(floor) - 4} fill="#8A93A1" fontSize="11" fontFamily="IBM Plex Mono">DRAWDOWN FLOOR</text>
        </g>
      )}
      <polyline points={pts} fill="none" stroke="#ECEEF0" strokeWidth="1.5" />
    </svg>
  );
}

export function LimitGauge({ used, label }) {
  const pct = Math.min(100, Math.max(0, used * 100));
  const color = pct >= 100 ? "#D64545" : pct >= 70 ? "#D4972B" : "#35B075";
  return (
    <div>
      <div className="caption">{label} {pct.toFixed(0)}% OF LIMIT</div>
      <div className="gauge"><div style={{ width: `${pct}%`, background: color }} /></div>
    </div>
  );
}
