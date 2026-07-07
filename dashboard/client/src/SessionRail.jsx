import React from "react";
import { usePoll } from "./api.js";

// Trading day mapped left to right: 17:00 ET (day open) to 17:00 ET next day.
// Segments from the live session calendar config, brightened where sessions
// overlap. Blue marker sweeps as simulated time advances.

const DAY_MIN = 24 * 60;

function toOffset(hhmm) {
  // minutes since 17:00 ET, the left edge of the rail
  const [h, m] = hhmm.split(":").map(Number);
  return (h * 60 + m - 17 * 60 + DAY_MIN) % DAY_MIN;
}

export default function SessionRail({ simTime, mode }) {
  const sessions = usePoll("/api/sessions", 60000);
  const fx = sessions?.EUR_USD?.sessions || [];

  // build overlap counts in 15-minute buckets
  const buckets = new Array(96).fill(0);
  const spans = fx.map((s) => {
    const a = toOffset(s.start_et);
    let b = toOffset(s.end_et);
    if (b <= a) b += DAY_MIN;
    for (let t = a; t < b; t += 15) buckets[Math.floor((t % DAY_MIN) / 15)]++;
    return { ...s, a, b };
  });

  let markerPct = null;
  if (simTime) {
    const d = new Date(simTime);
    const et = new Date(d.toLocaleString("en-US", { timeZone: "America/New_York" }));
    const mins = (et.getHours() * 60 + et.getMinutes() - 17 * 60 + DAY_MIN) % DAY_MIN;
    markerPct = (mins / DAY_MIN) * 100;
  }

  return (
    <div className="session-rail" title="Trading day, 17:00 ET to 17:00 ET">
      {spans.map((s) => (
        <div key={s.name} className="session-seg"
          style={{ left: `${(s.a / DAY_MIN) * 100}%`, width: `${((s.b - s.a) / DAY_MIN) * 100}%` }}>
          <span className="caption">{s.name.replace("_", " ")}</span>
        </div>
      ))}
      {buckets.map((n, i) => n > 1 && (
        <div key={i} className="session-seg overlap"
          style={{ left: `${(i / 96) * 100}%`, width: `${100 / 96}%`, borderLeft: "none" }} />
      ))}
      {markerPct != null && <div className="time-marker" style={{ left: `${markerPct}%` }} />}
      <span className={`tag blue mode-tag`}>{mode}</span>
    </div>
  );
}
