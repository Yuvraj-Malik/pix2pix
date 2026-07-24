import { useEffect, useState } from 'react';
import Header from './components/Header';
import UploadZone from './components/UploadZone';
import MetricsPanel from './components/MetricsPanel';
import DashboardGrid from './components/DashboardGrid';

const MAX_FILE_SIZE_MB = 10;
const ACCEPTED_TYPES = ['image/png', 'image/jpeg'];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState(null);
  const [uiError, setUiError] = useState('');
  const [analysisMeta, setAnalysisMeta] = useState(null);

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const resetCurrentPreview = () => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
  };

  const handleFileSelect = (file) => {
    if (!file) {
      return;
    }

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setUiError('Unsupported file type. Please upload a PNG or JPEG CT slice.');
      return;
    }

    const fileSizeMb = file.size / (1024 * 1024);
    if (fileSizeMb > MAX_FILE_SIZE_MB) {
      setUiError(`File exceeds ${MAX_FILE_SIZE_MB} MB. Please upload a smaller image.`);
      return;
    }

    setUiError('');
    resetCurrentPreview();
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setResults(null);
    setAnalysisMeta(null);
  };

  const handleClearFile = () => {
    resetCurrentPreview();
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    setAnalysisMeta(null);
    setUiError('');
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    const startedAt = Date.now();
    setIsProcessing(true);
    setUiError('');

    // Mock API Call - Replace with your Python backend fetch later
    setTimeout(() => {
      setResults({
        metrics: {
          dice: 0.7225,
          iou: 0.5656,
          precision: 0.6584,
          sensitivity: 0.8005,
          specificity: 0.8701
        }
      });
      setAnalysisMeta({
        runAt: new Date().toISOString(),
        elapsedMs: Date.now() - startedAt,
        modelName: 'Pix2Pix GAN (Epoch 100, frozen)'
      });
      setIsProcessing(false);
    }, 2500);
  };

  const modelFacts = [
    { label: 'CT Window', value: '[-100, 200] HU' },
    { label: 'Input Resolution', value: '256 x 256' },
    { label: 'PatchGAN Grid', value: '31 x 31' },
    { label: 'Hemorrhage Weight', value: '30x BCE' }
  ];

  return (
    <div className="app-shell">
      <div className="app-container">
        <Header analysisMeta={analysisMeta} isProcessing={isProcessing} />

        <section className="workflow-strip" aria-label="Workflow steps">
          <article className={`workflow-step ${selectedFile ? 'done' : 'active'}`}>
            <p className="workflow-index">1</p>
            <div>
              <p className="workflow-title">Upload CT Slice</p>
              <p className="workflow-copy">Select a single axial image for model input.</p>
            </div>
          </article>
          <article className={`workflow-step ${isProcessing ? 'active' : results ? 'done' : ''}`}>
            <p className="workflow-index">2</p>
            <div>
              <p className="workflow-title">Run Segmentation</p>
              <p className="workflow-copy">Model generates predicted mask and quality metrics.</p>
            </div>
          </article>
          <article className={`workflow-step ${results ? 'done' : ''}`}>
            <p className="workflow-index">3</p>
            <div>
              <p className="workflow-title">Clinical Review</p>
              <p className="workflow-copy">Compare raw input, output, and error context.</p>
            </div>
          </article>
        </section>

        <section className="fact-strip" aria-label="Model architecture details">
          {modelFacts.map((fact) => (
            <article key={fact.label} className="fact-card">
              <p className="fact-label">{fact.label}</p>
              <p className="fact-value">{fact.value}</p>
            </article>
          ))}
        </section>

        <main className="workspace-grid">
          <aside className="control-rail">
            <UploadZone
              onFileSelect={handleFileSelect}
              onClearFile={handleClearFile}
              previewUrl={previewUrl}
              isProcessing={isProcessing}
              onAnalyze={handleAnalyze}
              selectedFile={selectedFile}
              uiError={uiError}
            />
            <MetricsPanel
              metrics={results?.metrics}
              isProcessing={isProcessing}
              analysisMeta={analysisMeta}
            />
          </aside>

          <DashboardGrid
            previewUrl={previewUrl}
            results={results}
            isProcessing={isProcessing}
            selectedFile={selectedFile}
          />
        </main>

        <footer className="footer-note">
          For radiology triage support only. Final interpretation must remain with a licensed clinician.
        </footer>
      </div>
    </div>
  );
}