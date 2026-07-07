import { useEffect, useRef, useState } from "react";

export const get = (path) => fetch(path).then((r) => r.json());

// One WebSocket for the whole app; handlers keyed by table name.
export function useLiveFeed(handlers) {
  const ref = useRef(handlers);
  ref.current = handlers;
  useEffect(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onmessage = (e) => {
      const msg = JSON.parse(e.data);
      ref.current[msg.type]?.(msg.rows);
    };
    return () => ws.close();
  }, []);
}

export function usePoll(path, ms = 5000) {
  const [data, setData] = useState(null);
  useEffect(() => {
    let live = true;
    const load = () => get(path).then((d) => live && setData(d));
    load();
    const id = setInterval(load, ms);
    return () => { live = false; clearInterval(id); };
  }, [path, ms]);
  return data;
}

export const fmt = (n, dp = 2) =>
  n == null ? "—" : n.toLocaleString("en-US", { minimumFractionDigits: dp, maximumFractionDigits: dp });
