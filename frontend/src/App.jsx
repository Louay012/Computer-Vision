import React, { useState, useEffect, useRef } from 'react'

function Spinner() {
  return (
    <div className="spinner" aria-hidden>
      <div className="double-bounce1"></div>
      <div className="double-bounce2"></div>
    </div>
  )
}

function Header() {
  return (
    <header className="header">
      <div className="brand">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <rect x="1" y="1" width="22" height="22" rx="6" fill="#06b6d4" opacity="0.12" />
          <path d="M7 14c1-3 6-3 7 0" stroke="#06b6d4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          <circle cx="12" cy="8" r="2" fill="#10b981" />
        </svg>
        <div>
          <h1>Plant Disease</h1>
          <div className="subtitle">Image-based plant disease detection</div>
        </div>
      </div>
      <nav className="header-actions">
        <a className="link-btn" href="#" onClick={(e) => e.preventDefault()}>Docs</a>
      </nav>
    </header>
  )
}

export default function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [modelType, setModelType] = useState('dl')
  const [mlModels, setMlModels] = useState([])
  const [mlModel, setMlModel] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  const fetchModels = () => {
    fetch('http://localhost:8000/api/ml_models')
      .then(r => r.json())
      .then(d => { setMlModels(d.models || []); if (d.models && d.models.length && !mlModel) setMlModel(d.models[0]) })
      .catch(() => {})
  }

  useEffect(() => {
    fetchModels()
  }, [])

  useEffect(() => {
    if (!file) {
      setPreview(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const handleFileInput = (f) => {
    setFile(f)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files && e.dataTransfer.files[0]
    if (f) handleFileInput(f)
  }

  const submit = async (e) => {
    e?.preventDefault()
    if (!file) return
    setLoading(true)
    setResult(null)
    const form = new FormData()
    form.append('file', file)
    form.append('model_type', modelType)
    if (modelType === 'ml') form.append('model_name', mlModel)
    try {
      const res = await fetch('http://localhost:8000/api/predict', { method: 'POST', body: form })
      const json = await res.json()
      if (!res.ok) setResult({ error: json?.detail || JSON.stringify(json) })
      else setResult(json)
    } catch (err) {
      setResult({ error: err.message })
    } finally {
      setLoading(false)
    }
  }

  const refreshModels = () => {
    fetchModels()
  }

  return (
    <div className="app">
      <Header />

      <div className="form-grid">
        <form className="uploader-column" onSubmit={submit}>
          <div
            className={`uploader ${dragOver ? 'dragover' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => inputRef.current && inputRef.current.click()}
          >
            <input ref={inputRef} type="file" accept="image/*" style={{display: 'none'}} onChange={e => handleFileInput(e.target.files && e.target.files[0])} />
            {preview ? (
              <>
                <button type="button" className="remove-btn" onClick={(ev) => { ev.stopPropagation(); setFile(null); setPreview(null); setResult(null); if (inputRef.current) inputRef.current.value = null }} aria-label="Remove image">✕</button>
                <img src={preview} alt="preview" className="thumb" />
              </>
            ) : (
              <div className="upload-instructions">Drag & drop an image here, or click to select</div>
            )}
          </div>

          <div className="uploader-footer">
            {file ? <div className="filename">{file.name}</div> : <div className="helper">Supported: JPG, PNG — Max size: 10MB</div>}
          </div>

          <div className="controls">
            <label>Model type:</label>
            <select value={modelType} onChange={e => setModelType(e.target.value)}>
              <option value="dl">Deep Learning (ResNet)</option>
              <option value="ml">Classical (feature-based)</option>
            </select>

            {modelType === 'ml' && (
              <div className="ml-select-row">
                <label>ML model:</label>
                <select value={mlModel} onChange={e => setMlModel(e.target.value)}>
                  {mlModels.map(m => <option key={m} value={m}>{m}</option>)}
                </select>
                <button type="button" className="link-btn" onClick={refreshModels}>Refresh</button>
              </div>
            )}
          </div>

          <div className="actions">
            <button type="submit" disabled={loading || !file} className="primary">{loading ? 'Predicting...' : 'Predict'}</button>
            <button type="button" onClick={() => { setFile(null); setResult(null); if (inputRef.current) inputRef.current.value = null }} className="muted">Clear</button>
          </div>
        </form>

        <div className="preview-column">
          <div className="result-card">
            <h2>Result</h2>
            {loading ? (
              <div className="center"><Spinner /></div>
            ) : result ? (
              result.error ? (
                <div className="error">{result.error}</div>
              ) : (
                <div>
                  <div className="result-main">
                            <div className="prediction-badge">
                              <span className="label">Prediction</span>
                              <span className="prediction">{result.prediction || result.label || result[0] || '—'}</span>
                            </div>
                  </div>
                  {result.probability !== undefined && (
                    <div className="prob">
                      <div className="prob-bar">
                        <div className="prob-fill" style={{width: `${Math.round(result.probability*100)}%`}} />
                      </div>
                      <small>{Math.round(result.probability*100)}% confidence</small>
                    </div>
                  )}
                  <details className="raw">
                    <summary>Raw response</summary>
                    <pre>{JSON.stringify(result, null, 2)}</pre>
                  </details>
                </div>
              )
            ) : (
              <div className="empty">No result yet. Upload an image and choose a model.</div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
