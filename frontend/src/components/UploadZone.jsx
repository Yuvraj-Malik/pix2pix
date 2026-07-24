import { useRef, useState } from 'react';
import { FileImage, Trash2, UploadCloud } from 'lucide-react';

function formatFileSize(bytes) {
  if (!bytes) return '0 MB';
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function UploadZone({
  onFileSelect,
  onClearFile,
  previewUrl,
  isProcessing,
  onAnalyze,
  selectedFile,
  uiError
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const inputRef = useRef(null);

  const openFilePicker = () => {
    inputRef.current?.click();
  };

  const onInputChange = (event) => {
    const file = event.target.files?.[0];
    onFileSelect(file);
    event.target.value = '';
  };

  const onDrop = (event) => {
    event.preventDefault();
    setIsDragOver(false);
    const file = event.dataTransfer.files?.[0];
    onFileSelect(file);
  };

  return (
    <section className="card control-card">
      <div className="card-title-wrap">
        <h2 className="card-title">Case Input</h2>
        <span className="status-tag">Single-slice review</span>
      </div>

      {!previewUrl ? (
        <div
          className={`upload-dropzone ${isDragOver ? 'drag-over' : ''}`}
          role="button"
          tabIndex={0}
          onClick={openFilePicker}
          onKeyDown={(event) => {
            if (event.key === 'Enter' || event.key === ' ') {
              event.preventDefault();
              openFilePicker();
            }
          }}
          onDragOver={(event) => {
            event.preventDefault();
            setIsDragOver(true);
          }}
          onDragEnter={(event) => {
            event.preventDefault();
            setIsDragOver(true);
          }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={onDrop}
          aria-label="Upload CT slice"
        >
          <UploadCloud className="upload-icon" />
          <span className="upload-title">Upload CT slice</span>
          <span className="upload-note">Accepted formats: PNG, JPG. The image is normalized and resized to 256 x 256.</span>
          <button type="button" className="secondary-button" onClick={openFilePicker}>
            Select file
          </button>
          <input
            ref={inputRef}
            type="file"
            className="visually-hidden"
            accept="image/png, image/jpeg"
            onChange={onInputChange}
          />
        </div>
      ) : (
        <div className="upload-preview-wrap">
          <img src={previewUrl} alt="Uploaded CT slice" className="preview-image" />
          {selectedFile ? (
            <div className="file-meta" aria-label="Uploaded file details">
              <p className="file-name" title={selectedFile.name}>
                <FileImage size={15} aria-hidden="true" />
                <span>{selectedFile.name}</span>
              </p>
              <p className="file-size">{formatFileSize(selectedFile.size)}</p>
            </div>
          ) : null}

          <div className="action-row">
            <button
              onClick={onAnalyze}
              disabled={isProcessing}
              className="analyze-button"
              type="button"
            >
              {isProcessing ? 'Running segmentation model...' : 'Generate segmentation'}
            </button>

            <button
              type="button"
              className="danger-button"
              onClick={onClearFile}
              disabled={isProcessing}
            >
              <Trash2 size={14} aria-hidden="true" />
              Clear
            </button>
          </div>
        </div>
      )}

      {uiError ? <p className="error-text" role="alert">{uiError}</p> : null}
    </section>
  );
}