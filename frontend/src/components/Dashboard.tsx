"use client";

import { useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

type Metrics = {
  total_return: number;
  annualized_volatility: number;
  sharpe: number;
  max_drawdown: number;
};

type Run = {
  run_id: string;
  dataset_version: string;
  dataset_hash: string;
  mode: string;
  provider: string;
  model: string;
  decision_count: number;
  development_decisions: number;
  holdout_decisions: number;
  metrics: Record<string, Metrics>;
  partition_metrics?: Record<string, Record<string, Metrics>>;
  average_monthly_turnover?: Record<string, number>;
  routing?: Record<string, { provider: string; model: string }>;
  forecast_reconciliation_count?: number;
  matured_stock_forecast_outcomes?: number;
  forecast_quality?: {
    matured_forecasts: number;
    standalone_mae?: number;
    reconciled_mae?: number;
    standalone_directional_accuracy?: number;
    reconciled_directional_accuracy?: number;
  };
  transaction_costs_ignored?: boolean;
  warning: string;
};

type CurvePoint = { date: string; strategy: string; nav: number };
type Decision = {
  month_id: string;
  strategy_version: string;
  strategy_params: Record<string, number>;
  weights: Record<string, number>;
  accepted_experiment_id: string | null;
  accepted_projection_id: string;
  forecast_reconciliation_id?: string;
  forecast_agent_steps?: number;
  development: boolean;
};
type Experiment = {
  experiment_id: string;
  status: string;
  proposal?: { hypothesis: string; overrides: Record<string, number>; expected_tradeoffs: string[]; evidence_ids: string[] };
  inner_validation_score?: number;
  one_way_turnover?: number;
  projection_id?: string;
};
type Projection = {
  projection_id: string;
  experiment_id: string;
  expected_return_3m: number;
  annualized_volatility: number;
  p10: [string, number][];
  p50: [string, number][];
  p90: [string, number][];
};
type MonthDetail = {
  decision: Decision;
  experiments: Experiment[];
  projections: Projection[];
  outcomes: {
    projection_id: string;
    realized_63_session_return: number;
    forecast_error: number;
    status: string;
  }[];
  forecasts: {
    forecast_id: string;
    symbol: string;
    expected_return_3m: number;
    confidence: number;
  }[];
  analyses: {
    analysis_id: string;
    symbol: string;
    rationale: string;
    evidence_ids: string[];
    provider: string;
    model: string;
  }[];
  forecast_reconciliations: {
    reconciliation_id: string;
    market_regime: string;
    market_expected_return_3m: number;
    confidence: number;
    rationale: string;
    forecasts: {
      symbol: string;
      source_forecast_id: string;
      raw_expected_return_3m: number;
      expected_return_3m: number;
      confidence: number;
      relative_rank: number;
      bear_case_return: number;
      bull_case_return: number;
      adjustment_rationale: string;
    }[];
  }[];
  stock_forecast_outcomes: {
    source_forecast_id: string;
    symbol: string;
    realized_63_session_return: number;
    forecast_error: number;
    direction_correct: boolean;
  }[];
  agent_actions: {
    step: number;
    action: string;
    symbols: string[];
    question: string;
    rationale: string;
    expected_information_gain: number;
    status: string;
  }[];
  agent_observations: unknown[];
  traces: unknown[];
};
type Dataset = { path: string; dataset_version: string; profile: string };
type Job = { job_id: string; status: string; run_id?: string; error?: string };

const LABELS: Record<string, string> = {
  original_baseline: "Original baseline",
  development_selected_fixed: "Development-selected fixed",
  adaptive_agent: "Adaptive agent",
};
const COLORS: Record<string, string> = {
  original_baseline: "#7f8b99",
  development_selected_fixed: "#d49a37",
  adaptive_agent: "#35c8a1",
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API}${path}`);
  if (!response.ok) throw new Error(`${response.status}: ${response.statusText}`);
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`${response.status}: ${response.statusText}`);
  return response.json() as Promise<T>;
}

function formatPercent(value: number) {
  return new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 }).format(value);
}

function EquityChart({ points }: { points: CurvePoint[] }) {
  const width = 980;
  const height = 330;
  const pad = 34;
  const grouped = useMemo(() => {
    const map = new Map<string, CurvePoint[]>();
    for (const point of points) {
      const values = map.get(point.strategy) ?? [];
      values.push(point);
      map.set(point.strategy, values);
    }
    return map;
  }, [points]);
  const values = points.map((point) => point.nav);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 1);
  const paths = [...grouped].map(([strategy, series]) => {
    const path = series
      .map((point, index) => {
        const x = pad + (index / Math.max(series.length - 1, 1)) * (width - pad * 2);
        const y = height - pad - ((point.nav - min) / range) * (height - pad * 2);
        return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
    return <path key={strategy} d={path} fill="none" stroke={COLORS[strategy]} strokeWidth="2.4" />;
  });
  return (
    <div className="chartWrap">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Three realized equity curves">
        <line x1={pad} y1={height - pad} x2={width - pad} y2={height - pad} className="axis" />
        <line x1={pad} y1={pad} x2={pad} y2={height - pad} className="axis" />
        {paths}
        <text x={pad} y={22} className="chartLabel">${max.toLocaleString(undefined, { maximumFractionDigits: 0 })}</text>
        <text x={pad} y={height - 10} className="chartLabel">${min.toLocaleString(undefined, { maximumFractionDigits: 0 })}</text>
      </svg>
      <div className="legend">
        {[...grouped.keys()].map((strategy) => (
          <span key={strategy}><i style={{ background: COLORS[strategy] }} />{LABELS[strategy]}</span>
        ))}
      </div>
    </div>
  );
}

function ProjectionChart({ projection }: { projection?: Projection }) {
  if (!projection) return <div className="empty">No projection selected.</div>;
  const width = 560;
  const height = 210;
  const all = [...projection.p10, ...projection.p90].map(([, value]) => value);
  const min = Math.min(...all);
  const max = Math.max(...all);
  const xy = ([, value]: [string, number], index: number, length: number) => [
    18 + (index / Math.max(length - 1, 1)) * (width - 36),
    height - 18 - ((value - min) / Math.max(max - min, 1)) * (height - 36),
  ];
  const line = (series: [string, number][]) => series.map((value, index) => {
    const [x, y] = xy(value, index, series.length);
    return `${index === 0 ? "M" : "L"}${x},${y}`;
  }).join(" ");
  const upper = projection.p90.map((value, index) => xy(value, index, projection.p90.length));
  const lower = projection.p10.map((value, index) => xy(value, index, projection.p10.length)).reverse();
  const area = [...upper, ...lower].map(([x, y], index) => `${index === 0 ? "M" : "L"}${x},${y}`).join(" ") + " Z";
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Frozen three-month projection fan">
      <path d={area} className="fan" />
      <path d={line(projection.p50)} className="median" />
    </svg>
  );
}

export function Dashboard() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [run, setRun] = useState<Run | null>(null);
  const [curves, setCurves] = useState<CurvePoint[]>([]);
  const [months, setMonths] = useState<Decision[]>([]);
  const [detail, setDetail] = useState<MonthDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [dataset, setDataset] = useState("");
  const [mode, setMode] = useState("mock");
  const [provider, setProvider] = useState("openai");
  const [model, setModel] = useState("gpt-4.1-mini");
  const [job, setJob] = useState<Job | null>(null);

  const refreshRuns = () => getJson<Run[]>("/api/runs").then((values) => {
    setRuns(values);
    if (!run && values[0]) setRun(values[0]);
    return values;
  });

  useEffect(() => {
    Promise.all([refreshRuns(), getJson<Dataset[]>("/api/datasets")]).then(([, values]) => {
      setDatasets(values);
      if (values[0]) setDataset(values[0].path);
    }).catch((reason: Error) => setError(reason.message));
    // Initial load intentionally runs once; later refreshes are explicit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!job || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(() => {
      getJson<Job>(`/api/jobs/${job.job_id}`).then((value) => {
        setJob(value);
        if (value.status === "complete") {
          refreshRuns().then((values) => {
            const created = values.find((item) => item.run_id === value.run_id);
            if (created) setRun(created);
          });
        }
      }).catch((reason: Error) => setError(reason.message));
    }, 1500);
    return () => window.clearInterval(timer);
  });

  useEffect(() => {
    if (!run) return;
    Promise.all([
      getJson<CurvePoint[]>(`/api/runs/${run.run_id}/equity-curves`),
      getJson<Decision[]>(`/api/runs/${run.run_id}/months`),
    ]).then(([curveValues, monthValues]) => {
      setCurves(curveValues);
      setMonths(monthValues);
      const selected = monthValues.at(-1);
      if (selected) getJson<MonthDetail>(`/api/runs/${run.run_id}/months/${selected.month_id}`).then(setDetail);
    }).catch((reason: Error) => setError(reason.message));
  }, [run]);

  const accepted = detail?.experiments.find((item) => item.status === "accepted");
  const projection = detail?.projections.find(
    (item) => item.projection_id === detail.decision.accepted_projection_id,
  );
  const outcome = detail?.outcomes.find(
    (item) => item.projection_id === detail.decision.accepted_projection_id,
  );
  const reconciliation = detail?.forecast_reconciliations.find(
    (item) => item.reconciliation_id === detail.decision.forecast_reconciliation_id,
  ) ?? detail?.forecast_reconciliations.at(-1);
  const stockOutcomeByForecast = new Map(
    (detail?.stock_forecast_outcomes ?? []).map((item) => [item.source_forecast_id, item]),
  );

  const startRun = () => {
    setError(null);
    postJson<Job>("/api/runs", {
      dataset,
      mode,
      provider: mode === "live" ? provider : null,
      model: mode === "live" ? model : null,
      iterations: 3,
    }).then(setJob).catch((reason: Error) => setError(reason.message));
  };

  return (
    <main>
      <header>
        <div><span className="eyebrow">POINT-IN-TIME RESEARCH SYSTEM</span><h1>Portfolio Agent</h1></div>
        <div className="runControls">
          <div className="runControl"><label htmlFor="run">Completed run</label><select id="run" value={run?.run_id ?? ""} onChange={(event) => setRun(runs.find((item) => item.run_id === event.target.value) ?? null)}>{runs.map((item) => <option key={item.run_id} value={item.run_id}>{item.run_id}</option>)}</select></div>
          <div className="runControl"><label htmlFor="dataset">Dataset</label><select id="dataset" value={dataset} onChange={(event) => setDataset(event.target.value)}>{datasets.map((item) => <option key={item.path} value={item.path}>{item.dataset_version}</option>)}</select></div>
          <div className="runControl"><label htmlFor="mode">Inference</label><select id="mode" value={mode} onChange={(event) => setMode(event.target.value)}><option value="mock">Offline mock</option><option value="live">Live provider</option></select></div>
          {mode === "live" ? <><div className="runControl"><label htmlFor="provider">Provider</label><select id="provider" value={provider} onChange={(event) => setProvider(event.target.value)}><option value="openai">OpenAI</option><option value="anthropic">Claude</option><option value="gemini">Gemini</option><option value="muse">Muse</option></select></div><div className="runControl"><label htmlFor="model">Model ID</label><input id="model" value={model} onChange={(event) => setModel(event.target.value)} /></div></> : null}
          <button className="runButton" onClick={startRun} disabled={!dataset || job?.status === "running" || job?.status === "queued"}>{job && ["queued", "running"].includes(job.status) ? job.status : "Run replay"}</button>
        </div>
      </header>

      {error ? <div className="error">API unavailable: {error}</div> : null}
      {!run ? <div className="loading">Waiting for a completed replay…</div> : (
        <>
          <section className="provenance">
            <div><span>Dataset</span><strong>{run.dataset_version}</strong><small>{run.dataset_hash.slice(0, 16)}…</small></div>
            <div><span>Inference</span><strong>{run.provider} / {run.model}</strong><small>{Object.keys(run.routing ?? {}).length ? Object.entries(run.routing ?? {}).map(([stage, route]) => `${stage}: ${route.provider}/${route.model}`).join(" · ") : `${run.mode} mode · default route`}</small></div>
            <div><span>Decisions</span><strong>{run.decision_count} months</strong><small>{run.development_decisions} development · {run.holdout_decisions} holdout</small></div>
            <div className="warning"><span>Provenance</span><strong>Synthetic · retrospective</strong><small>Not investment advice</small></div>
          </section>

          <section className="panel heroPanel">
            <div className="panelTitle"><div><span className="eyebrow">REALIZED · NO TRANSACTION COSTS</span><h2>Portfolio paths</h2></div><span className="pill">Common calendar</span></div>
            <EquityChart points={curves} />
            <div className="metricsGrid">
              {Object.entries(run.metrics).map(([strategy, metric]) => (
                <article key={strategy}>
                  <h3><i style={{ background: COLORS[strategy] }} />{LABELS[strategy]}</h3>
                  <dl><div><dt>Return</dt><dd>{formatPercent(metric.total_return)}</dd></div><div><dt>Sharpe</dt><dd>{metric.sharpe.toFixed(2)}</dd></div><div><dt>Drawdown</dt><dd>{formatPercent(metric.max_drawdown)}</dd></div><div><dt>Volatility</dt><dd>{formatPercent(metric.annualized_volatility)}</dd></div><div><dt>Avg turnover</dt><dd>{formatPercent(run.average_monthly_turnover?.[strategy] ?? 0)}</dd></div></dl>
                </article>
              ))}
            </div>
          </section>

          <section className="panel partitionPanel">
            <div className="panelTitle"><div><span className="eyebrow">SEALED EVALUATION</span><h2>Development vs holdout</h2></div><span className="pill">Costs ignored · same calendar</span></div>
            <div className="partitionGrid">{Object.entries(run.partition_metrics ?? {}).map(([partition, strategies]) => <article key={partition}><h3>{partition}</h3>{Object.entries(strategies).map(([strategy, metric]) => <div key={strategy}><span>{LABELS[strategy]}</span><b>{formatPercent(metric.total_return)}</b><small>Sharpe {metric.sharpe.toFixed(2)}</small></div>)}</article>)}</div>
          </section>

          {run.forecast_quality && run.forecast_quality.matured_forecasts > 0 ? (
            <section className="panel forecastQualityPanel">
              <div className="panelTitle">
                <div><span className="eyebrow">FORECAST EVALUATION</span><h2>Did reconciliation help?</h2></div>
                <span className="pill">{run.forecast_quality.matured_forecasts} matured company forecasts</span>
              </div>
              <div className="qualityGrid">
                <div><span>Standalone MAE</span><strong>{formatPercent(run.forecast_quality.standalone_mae ?? 0)}</strong></div>
                <div><span>Reconciled MAE</span><strong>{formatPercent(run.forecast_quality.reconciled_mae ?? 0)}</strong></div>
                <div><span>Standalone direction</span><strong>{formatPercent(run.forecast_quality.standalone_directional_accuracy ?? 0)}</strong></div>
                <div><span>Reconciled direction</span><strong>{formatPercent(run.forecast_quality.reconciled_directional_accuracy ?? 0)}</strong></div>
              </div>
            </section>
          ) : null}

          <section className="workspace">
            <aside className="panel timeline">
              <div className="panelTitle"><div><span className="eyebrow">CHRONOLOGY</span><h2>Monthly decisions</h2></div></div>
              <div className="monthList">
                {months.map((month, index) => (
                  <button key={month.month_id} className={detail?.decision.month_id === month.month_id ? "active" : ""} onClick={() => getJson<MonthDetail>(`/api/runs/${run.run_id}/months/${month.month_id}`).then(setDetail)}>
                    <span>{month.month_id.slice(0, 7)}</span><small>{month.development ? "Development" : "Holdout"}</small><b>{String(index + 1).padStart(2, "0")}</b>
                  </button>
                ))}
              </div>
            </aside>

            <div className="detailColumn">
              <section className="panel">
                <div className="panelTitle"><div><span className="eyebrow">FROZEN AT DECISION TIME</span><h2>Three-month projection</h2></div>{projection ? <span className="pill">{formatPercent(projection.expected_return_3m)} expected</span> : null}</div>
                <ProjectionChart projection={projection} />
                <div className="predictionActual">
                  <div><span>Prediction</span><strong>{projection ? formatPercent(projection.expected_return_3m) : "—"}</strong></div>
                  <div><span>Actual after 63 sessions</span><strong>{outcome ? formatPercent(outcome.realized_63_session_return) : "Pending"}</strong></div>
                  <div><span>Forecast error</span><strong>{outcome ? formatPercent(outcome.forecast_error) : "—"}</strong></div>
                </div>
              </section>
              <section className="panel allocations">
                <div className="panelTitle"><div><span className="eyebrow">ACCEPTED PORTFOLIO</span><h2>Ten-stock allocation</h2></div><span className="hash">{detail?.decision.strategy_version}</span></div>
                <div className="bars">
                  {Object.entries(detail?.decision.weights ?? {}).sort((a, b) => b[1] - a[1]).map(([symbol, weight]) => (
                    <div key={symbol}><span>{symbol}</span><div><i style={{ width: `${weight * 400}%` }} /></div><b>{formatPercent(weight)}</b></div>
                  ))}
                </div>
              </section>
            </div>
          </section>

          <section className="panel agentPanel">
            <div className="panelTitle">
              <div>
                <span className="eyebrow">DECIDE → ACT → OBSERVE → REVISE</span>
                <h2>Forecasting agent trace</h2>
              </div>
              <span className="pill">{detail?.agent_actions.length ?? 0} decisions</span>
            </div>
            {(detail?.agent_actions.length ?? 0) === 0 ? (
              <div className="empty">This older run predates the dynamic forecasting agent.</div>
            ) : (
              <div className="actionTimeline">
                {detail?.agent_actions.map((action) => (
                  <article key={`${action.step}-${action.action}`}>
                    <b>{String(action.step + 1).padStart(2, "0")}</b>
                    <div>
                      <span>{action.action.replaceAll("_", " ")}</span>
                      <p>{action.question}</p>
                      <small>
                        {action.symbols.length ? `${action.symbols.join(", ")} · ` : ""}
                        expected information gain {formatPercent(action.expected_information_gain)}
                      </small>
                    </div>
                    <em>{action.status}</em>
                  </article>
                ))}
              </div>
            )}
          </section>

          <section className="panel reconciliationPanel">
            <div className="panelTitle">
              <div>
                <span className="eyebrow">CROSS-COMPANY CONSISTENCY</span>
                <h2>Reconciled return forecasts</h2>
              </div>
              {reconciliation ? (
                <span className="pill">
                  {reconciliation.market_regime} · {formatPercent(reconciliation.confidence)} confidence
                </span>
              ) : null}
            </div>
            {!reconciliation ? (
              <div className="empty">No reconciliation artifact exists for this run.</div>
            ) : (
              <>
                <p className="reconciliationRationale">{reconciliation.rationale}</p>
                <div className="forecastTableWrap">
                  <table className="forecastTable">
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Company</th>
                        <th>Standalone</th>
                        <th>Reconciled</th>
                        <th>Bear / bull</th>
                        <th>Confidence</th>
                        <th>Realized</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...reconciliation.forecasts]
                        .sort((a, b) => a.relative_rank - b.relative_rank)
                        .map((forecast) => {
                          const realized = stockOutcomeByForecast.get(forecast.source_forecast_id);
                          return (
                            <tr key={forecast.symbol}>
                              <td>{forecast.relative_rank}</td>
                              <td><strong>{forecast.symbol}</strong></td>
                              <td>{formatPercent(forecast.raw_expected_return_3m)}</td>
                              <td>{formatPercent(forecast.expected_return_3m)}</td>
                              <td>
                                {formatPercent(forecast.bear_case_return)} / {formatPercent(forecast.bull_case_return)}
                              </td>
                              <td>{formatPercent(forecast.confidence)}</td>
                              <td>{realized ? formatPercent(realized.realized_63_session_return) : "Pending"}</td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>

          <section className="panel experiments">
            <div className="panelTitle"><div><span className="eyebrow">MEASURED, NOT NARRATED</span><h2>Within-month experiments</h2></div></div>
            <div className="experimentGrid">
              {(detail?.experiments ?? []).filter((item) => item.proposal).map((item) => (
                <article key={`${item.experiment_id}-${item.status}`} className={item.status}>
                  <div><span className="status">{item.status}</span><small>{item.experiment_id}</small></div>
                  <h3>{item.proposal?.hypothesis}</h3>
                  <code>{JSON.stringify(item.proposal?.overrides)}</code>
                  <small className="evidence">Evidence: {item.proposal?.evidence_ids?.slice(0, 3).join(", ") || "none"}</small>
                  <dl><div><dt>Validation score</dt><dd>{item.inner_validation_score?.toFixed(3) ?? "—"}</dd></div><div><dt>Turnover</dt><dd>{item.one_way_turnover == null ? "—" : formatPercent(item.one_way_turnover)}</dd></div></dl>
                </article>
              ))}
            </div>
          </section>

          <section className="panel evidencePanel">
            <div className="panelTitle"><div><span className="eyebrow">DATED EVIDENCE</span><h2>Grounded company research</h2></div><span className="pill">{detail?.analyses.length ?? 0} analyses</span></div>
            <div className="evidenceGrid">{(detail?.analyses ?? []).slice(0, 6).map((analysis) => <article key={analysis.analysis_id}><div><strong>{analysis.symbol}</strong><small>{analysis.provider} / {analysis.model}</small></div><p>{analysis.rationale}</p><code>{analysis.evidence_ids.slice(0, 4).join(" · ") || "No dated evidence"}</code></article>)}</div>
          </section>

          <footer><p>{run.warning}</p><p>All model evidence is dated and all projections are immutable artifacts.</p></footer>
        </>
      )}
    </main>
  );
}
