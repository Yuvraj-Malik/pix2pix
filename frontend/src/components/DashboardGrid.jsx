import { AlertCircle, CheckCircle } from 'lucide-react';

export default function DashboardGrid({ previewUrl, results, isProcessing, selectedFile }) {
  return (
    <section className="card dashboard-card">
      <div className="card-title-wrap">
        <h2 className="card-title">Diagnostic Review Panel</h2>
        <span className="status-tag">Four-view comparison</span>
      </div>

      {results ? (
        <div className="diagnostic-grid">
          <div className="diagnostic-tile">
            <h3>Raw Input</h3>
            <img src={previewUrl} alt="Raw CT input" className="tile-image" />
          </div>

          <div className="diagnostic-tile">
            <h3>Ground Truth</h3>
            <div className="tile-placeholder">
              Ground-truth mask pending annotation
            </div>
          </div>

          <div className="diagnostic-tile">
            <h3>GAN Output</h3>
            <div className="tile-placeholder tile-positive">
              <CheckCircle size={34} />
              <span>Predicted segmentation mask</span>
            </div>
          </div>

          <div className="diagnostic-tile">
            <h3>Error Map</h3>
            <div className="tile-placeholder tile-alert">
              <AlertCircle size={34} />
              <span>Pixel-wise disagreement map</span>
            </div>
          </div>
        </div>
      ) : (
        <div className="dashboard-empty">
          {isProcessing ? (
            <span className="processing-text">Generating prediction and error map...</span>
          ) : (
            <>
              <p className="empty-title">Awaiting case input</p>
              <p className="empty-subtitle">Upload a CT slice to review model output against reference and error distribution.</p>
            </>
          )}
        </div>
      )}

      <section className="legend-box" aria-label="Interpretation legend">
        <h3>Interpretation Guide</h3>
        <p><strong>True positive:</strong> hemorrhage pixels correctly identified.</p>
        <p><strong>False positive:</strong> model highlights non-hemorrhage tissue.</p>
        <p><strong>False negative:</strong> hemorrhage pixels missed by the model.</p>
        {selectedFile ? (
          <p className="legend-case">Active case: {selectedFile.name}</p>
        ) : null}
      </section>
    </section>
  );
}