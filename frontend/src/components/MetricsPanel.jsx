const METRIC_CONTENT = [
  {
    key: 'dice',
    label: 'Dice (F1)',
    hint: 'Overall mask overlap'
  },
  {
    key: 'iou',
    label: 'IoU (Jaccard)',
    hint: 'Intersection consistency'
  },
  {
    key: 'precision',
    label: 'Precision',
    hint: 'False-positive control'
  },
  {
    key: 'sensitivity',
    label: 'Sensitivity',
    hint: 'Hemorrhage capture rate'
  },
  {
    key: 'specificity',
    label: 'Specificity',
    hint: 'Healthy tissue rejection'
  }
];

function getMetricBand(key, value) {
  const thresholds = {
    dice: [0.7, 0.8],
    iou: [0.55, 0.65],
    precision: [0.6, 0.72],
    sensitivity: [0.78, 0.86],
    specificity: [0.84, 0.9]
  };

  const [moderate, strong] = thresholds[key] || [0.5, 0.75];
  if (value >= strong) return 'strong';
  if (value >= moderate) return 'moderate';
  return 'watch';
}

export default function MetricsPanel({ metrics, isProcessing, analysisMeta }) {
  const hasMetrics = Boolean(metrics);

  return (
    <section className="card metric-card">
      <div className="card-title-wrap">
        <h2 className="card-title">Quantitative Results</h2>
        <span className="status-tag">Validation summary</span>
      </div>

      {isProcessing && !hasMetrics ? (
        <div className="metric-empty">Computing overlap and detection quality metrics...</div>
      ) : null}

      {!isProcessing && !hasMetrics ? (
        <div className="metric-empty">Run segmentation to display Dice, IoU, precision, sensitivity, and specificity.</div>
      ) : null}

      {hasMetrics ? (
        <div className="metric-grid">
          {METRIC_CONTENT.map((item) => (
            <article key={item.key} className="metric-row">
              <div>
                <p className="metric-label">{item.label}</p>
                <p className="metric-hint">{item.hint}</p>
              </div>
              <div className="metric-value-wrap">
                <p className="metric-value">{metrics[item.key].toFixed(4)}</p>
                <span className={`metric-band ${getMetricBand(item.key, metrics[item.key])}`}>
                  {getMetricBand(item.key, metrics[item.key])}
                </span>
              </div>
            </article>
          ))}
        </div>
      ) : null}

      {analysisMeta ? (
        <div className="run-meta" aria-label="Most recent run metadata">
          <p>Runtime: {(analysisMeta.elapsedMs / 1000).toFixed(2)}s</p>
          <p>Timestamp: {new Date(analysisMeta.runAt).toLocaleTimeString()}</p>
        </div>
      ) : null}
    </section>
  );
}