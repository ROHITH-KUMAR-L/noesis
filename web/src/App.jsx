import { useEffect, useState } from "react";
import { api } from "./lib.js";
import Playground from "./views/Playground.jsx";
import Replay from "./views/Replay.jsx";
import Results from "./views/Results.jsx";

const VIEWS = [
  ["replay", "Episode replay"],
  ["playground", "Cache playground"],
  ["results", "Results"],
];

export default function App() {
  const [view, setView] = useState(() => {
    const hash = window.location.hash.slice(1);
    return VIEWS.some(([id]) => id === hash) ? hash : "replay";
  });
  const [meta, setMeta] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api("/api/meta").then(setMeta).catch((e) => setError(e.message));
  }, []);

  const go = (id) => {
    setView(id);
    window.history.replaceState(null, "", `#${id}`);
  };

  return (
    <div className="frame">
      <header className="masthead">
        <div>
          <h1 className="wordmark">Noesis</h1>
          <p className="tagline">
            A learned memory controller for LLM agents that decides what to keep, and when to edit, by the real bill after
            prompt caching.
          </p>
        </div>
        <nav className="tabs" role="tablist" aria-label="Views">
          {VIEWS.map(([id, label]) => (
            <button key={id} type="button" role="tab" className="tab" aria-selected={view === id} onClick={() => go(id)}>
              {label}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {error ? (
          <p className="error">
            Cannot reach the demo API ({error}). Start it with <code>python -m memory_rl.web</code>.
          </p>
        ) : !meta ? (
          <p className="empty">Connecting to the demo API…</p>
        ) : view === "replay" ? (
          <Replay meta={meta} />
        ) : view === "playground" ? (
          <Playground meta={meta} />
        ) : (
          <Results />
        )}
      </main>

      <footer className="honesty">
        Everything here runs live: the KeyHunt toy tool environment, a scripted mock agent standing in for the frozen
        LLM, a prefix-cache cost simulator with recorded provider prices, and the trained PPO checkpoints. These are toy
        and mock-LLM results, not ALFWorld results and not measured provider bills.
      </footer>
    </div>
  );
}
