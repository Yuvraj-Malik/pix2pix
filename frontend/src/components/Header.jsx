import { Activity, Clock3, ShieldCheck } from 'lucide-react';

function formatRunTime(iso) {
  if (!iso) {
    return 'Not run yet';
  }

  return new Date(iso).toLocaleString();
}

export default function Header({ analysisMeta, isProcessing }) {
  return (
    <header className="hero-header">
      <div className="hero-left">
        <p className="eyebrow">Clinical Decision Support Interface</p>
        <h1>
          <Activity className="hero-title-icon" aria-hidden="true" />
          Intracranial Hemorrhage Segmentation Dashboard
        </h1>
        <p className="hero-subtitle">
          Pix2Pix-based CT segmentation with recall-prioritized detection to support emergency radiology triage.
        </p>
      </div>

      <div className="hero-right" aria-label="Pipeline highlights">
        <div className="hero-pill">
          <Activity size={16} aria-hidden="true" />
          Status: {isProcessing ? 'Running analysis' : 'Ready'}
        </div>
        <div className="hero-pill">
          <ShieldCheck size={16} aria-hidden="true" />
          {analysisMeta?.modelName || 'Pix2Pix model loaded'}
        </div>
        <div className="hero-pill">
          <Clock3 size={16} aria-hidden="true" />
          Last run: {formatRunTime(analysisMeta?.runAt)}
        </div>
      </div>
    </header>
  );
}