import { useState, useRef, useCallback } from 'react'
import axios from 'axios'
import {
  Upload,
  Activity,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Heart,
  FileText,
  Shield,
  TrendingUp,
  X,
  Sparkles,
  Info,
  RefreshCw,
  Brain,
  Scan
} from 'lucide-react'
import './App.css'

// Use relative URL in production (proxied by nginx), absolute in development
const API_BASE = import.meta.env.DEV
  ? 'http://localhost:8001/api/v1'
  : '/api/v1'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [error, setError] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [cancerType, setCancerType] = useState('')  // '' = auto-detect
  const fileInputRef = useRef(null)

  const handleDrag = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }, [])

  const handleFile = (selectedFile) => {
    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setPrediction(null)
      setReport(null)
      setError(null)
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const openFilePicker = (e) => {
    e.stopPropagation()
    fileInputRef.current?.click()
  }

  const clearFile = (e) => {
    e.stopPropagation()
    setFile(null)
    setPreview(null)
    setPrediction(null)
    setReport(null)
    setError(null)
    if (fileInputRef.current) fileInputRef.current.value = ''
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setPrediction(null)
    setReport(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      // Pass cancer_type parameter if user selected one (not auto-detect)
      const url = cancerType ? `${API_BASE}/predict?cancer_type=${cancerType}` : `${API_BASE}/predict`
      const response = await axios.post(url, formData)
      setPrediction(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Analysis failed. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleGenerateReport = async () => {
    if (!prediction?.prediction) return
    setReportLoading(true)

    try {
      const response = await axios.post(`${API_BASE}/generate_report`, {
        cancer_type: prediction.prediction.class,
        cancer_name: prediction.prediction.name,
        category: prediction.prediction.category,
        confidence: prediction.prediction.confidence,
        severity: prediction.prediction.severity
      })
      setReport(response.data)
    } catch (err) {
      console.error('Report generation failed:', err)
    } finally {
      setReportLoading(false)
    }
  }

  // Determine which type of cancer was detected
  const isBrainCancer = prediction?.detected_image_type === 'brain_mri'
  const isBreastCancer = prediction?.detected_image_type === 'breast_histopathology'
  const isCervicalCancer = prediction?.detected_image_type === 'cervical_cytology'
  const isKidneyCancer = prediction?.detected_image_type === 'kidney_scan'
  const isLungCancer = prediction?.detected_image_type === 'lung_scan'
  const isColonCancer = prediction?.detected_image_type === 'colon_histopathology'
  const isLymphoma = prediction?.detected_image_type === 'lymphoma_histopathology'
  const isOralCancer = prediction?.detected_image_type === 'oral_histopathology'

  // Determine result styling based on severity
  const getResultClass = () => {
    if (!prediction?.prediction?.severity) return 'benign'
    const severity = prediction.prediction.severity.toLowerCase()
    if (severity === 'high') return 'malignant'
    if (severity === 'medium') return 'medium'
    return 'benign'
  }

  const resultClass = getResultClass()

  return (
    <div className={`app ${isBrainCancer ? 'brain-mode' : ''}`}>
      {/* Animated Background */}
      <div className="bg-animation">
        <div className={`bg-gradient ${isBrainCancer ? 'brain' : ''}`}></div>
        <div className="bg-pattern"></div>
      </div>

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon unified">
              <Scan className="scan-icon" />
              <Sparkles className="sparkle" />
            </div>
            <div className="logo-text">
              <h1>OncoAI Fusion</h1>
              <span>Multi-Cancer Detection System</span>
            </div>
          </div>
          <div className="header-stats">
            <div className="stat">
              <Shield size={18} />
              <span>AI-Powered</span>
            </div>
            <div className="stat">
              <Activity size={18} />
              <span>Auto-Detection</span>
            </div>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="hero-section">
          <h2>Intelligent Multi-Cancer Screening</h2>
          <p>Upload a medical image and select the cancer type for accurate analysis</p>
          <div className="supported-types">
            <span className="type-badge"><Heart size={14} /> Breast Histopathology</span>
            <span className="type-badge"><Brain size={14} /> Brain MRI</span>
            <span className="type-badge"><Scan size={14} /> Cervical Cytology</span>
            <span className="type-badge"><Activity size={14} /> Kidney Scan</span>
            <span className="type-badge"><Activity size={14} /> Lung Scan</span>
            <span className="type-badge"><Activity size={14} /> Colon Histopathology</span>
            <span className="type-badge"><Activity size={14} /> Lymphoma</span>
            <span className="type-badge"><Activity size={14} /> Oral Cancer</span>
          </div>
        </div>

        <div className="content-grid">
          {/* Upload Section */}
          <section className="upload-card glass-card">
            <div className="card-header">
              <Upload size={22} />
              <h3>Upload Medical Image</h3>
            </div>

            <div
              className={`dropzone ${dragActive ? 'active' : ''} ${preview ? 'has-image' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => !preview && fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                onChange={handleFileChange}
                accept="image/*"
                hidden
              />

              {preview ? (
                <div className="preview-wrapper">
                  <img src={preview} alt="Preview" className="preview-img" />
                  <div className="preview-actions">
                    <button className="action-btn change-btn" onClick={openFilePicker}>
                      <RefreshCw size={16} />
                      <span>Change</span>
                    </button>
                    <button className="action-btn remove-btn" onClick={clearFile}>
                      <X size={16} />
                      <span>Remove</span>
                    </button>
                  </div>
                </div>
              ) : (
                <div className="upload-content">
                  <div className="upload-icon-wrapper unified">
                    <Upload size={40} />
                  </div>
                  <h4>Drop image here</h4>
                  <p>or click to browse</p>
                  <span className="file-hint">Supports JPG, PNG, BMP • Brain, Breast, Cervical, Kidney, Lung, Colon, Lymphoma, or Oral</span>
                </div>
              )}
            </div>

            {/* Cancer Type Selector */}
            <div className="cancer-type-selector">
              <label>Cancer Type:</label>
              <select value={cancerType} onChange={(e) => setCancerType(e.target.value)}>
                <option value="">Auto-detect (less reliable)</option>
                <option value="brain">Brain MRI</option>
                <option value="breast">Breast Histopathology</option>
                <option value="cervical">Cervical Cytology</option>
                <option value="kidney">Kidney Scan</option>
                <option value="lung">Lung Scan</option>
                <option value="colon">Colon Histopathology</option>
                <option value="lymphoma">Lymphoma</option>
                <option value="oral">Oral Cancer</option>
              </select>
            </div>

            <button
              className={`analyze-btn unified ${loading ? 'loading' : ''} ${!file ? 'disabled' : ''}`}
              onClick={handleAnalyze}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <Loader2 size={20} className="spinner" />
                  <span>Analyzing & Detecting Type...</span>
                </>
              ) : (
                <>
                  <Scan size={20} />
                  <span>Analyze Image</span>
                </>
              )}
            </button>

            {error && (
              <div className="error-alert">
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}
          </section>

          {/* Results Section */}
          <section className="results-card glass-card">
            <div className="card-header">
              <FileText size={22} />
              <h3>Analysis Results</h3>
            </div>

            {!prediction ? (
              <div className="empty-results">
                <div className="empty-icon unified">
                  <TrendingUp size={48} />
                </div>
                <h4>Awaiting Analysis</h4>
                <p>Upload a medical image to receive AI-powered cancer screening results</p>
                <div className="features">
                  <div className="feature">
                    <CheckCircle2 size={16} />
                    <span>Auto Image Detection</span>
                  </div>
                  <div className="feature">
                    <CheckCircle2 size={16} />
                    <span>Brain, Breast, Cervical, Kidney, Lung, Colon & Lymphoma</span>
                  </div>
                  <div className="feature">
                    <CheckCircle2 size={16} />
                    <span>Instant Results</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="results-content">
                {/* Detected Type Badge */}
                <div className={`detected-type-badge ${isBrainCancer ? 'brain' : isCervicalCancer ? 'cervical' : isKidneyCancer ? 'kidney' : isLungCancer ? 'lung' : isColonCancer ? 'colon' : isLymphoma ? 'lymphoma' : isOralCancer ? 'oral' : 'breast'}`}>
                  {isBrainCancer ? <Brain size={16} /> : isCervicalCancer ? <Scan size={16} /> : isKidneyCancer ? <Activity size={16} /> : isLungCancer ? <Activity size={16} /> : isColonCancer ? <Activity size={16} /> : isLymphoma ? <Activity size={16} /> : isOralCancer ? <Activity size={16} /> : <Heart size={16} />}
                  <span>Detected: {isBrainCancer ? 'Brain MRI' : isCervicalCancer ? 'Cervical Cytology' : isKidneyCancer ? 'Kidney Scan' : isLungCancer ? 'Lung Scan' : isColonCancer ? 'Colon Histopathology' : isLymphoma ? 'Lymphoma' : isOralCancer ? 'Oral Cancer' : 'Breast Histopathology'}</span>
                </div>

                {/* Status Badge */}
                <div className={`status-badge ${resultClass}`}>
                  <CheckCircle2 size={18} />
                  <span>Analysis Complete</span>
                </div>

                {/* Main Result Card */}
                <div className={`result-main ${resultClass}`}>
                  <div className="result-icon">
                    {isBrainCancer ? <Brain size={36} /> : isCervicalCancer ? <Scan size={36} /> : isKidneyCancer ? <Activity size={36} /> : isLungCancer ? <Activity size={36} /> : isColonCancer ? <Activity size={36} /> : isLymphoma ? <Activity size={36} /> : isOralCancer ? <Activity size={36} /> : <Heart size={36} />}
                  </div>
                  <div className="result-info">
                    <span className="result-label">Diagnosis</span>
                    <h3 className="result-name">{prediction.prediction.name}</h3>
                    <div className={`severity-tag ${prediction.prediction.severity.toLowerCase()}`}>
                      {prediction.prediction.severity} Risk
                    </div>
                  </div>
                  <div className="confidence-circle">
                    <svg viewBox="0 0 100 100">
                      <circle className="bg" cx="50" cy="50" r="42" />
                      <circle
                        className="progress"
                        cx="50" cy="50" r="42"
                        style={{
                          strokeDasharray: `${prediction.prediction.confidence * 264} 264`
                        }}
                      />
                    </svg>
                    <div className="confidence-text">
                      <span className="value">{(prediction.prediction.confidence * 100).toFixed(0)}</span>
                      <span className="percent">%</span>
                    </div>
                  </div>
                </div>

                {/* Probability Bars */}
                {prediction.probabilities && (
                  <div className="probability-section">
                    <h4>Probability Distribution</h4>
                    <div className="prob-bars">
                      {isBreastCancer ? (
                        <>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label benign-text">Benign</span>
                              <span className="prob-value">{(prediction.probabilities.benign * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar benign" style={{ width: `${prediction.probabilities.benign * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label malignant-text">Malignant</span>
                              <span className="prob-value">{(prediction.probabilities.malignant * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar malignant" style={{ width: `${prediction.probabilities.malignant * 100}%` }} />
                            </div>
                          </div>
                        </>
                      ) : isCervicalCancer ? (
                        <>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label">Dyskeratotic</span>
                              <span className="prob-value">{(prediction.probabilities.dyskeratotic * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar cervical" style={{ width: `${prediction.probabilities.dyskeratotic * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label">Koilocytotic</span>
                              <span className="prob-value">{(prediction.probabilities.koilocytotic * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar cervical" style={{ width: `${prediction.probabilities.koilocytotic * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label">Metaplastic</span>
                              <span className="prob-value">{(prediction.probabilities.metaplastic * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar cervical-low" style={{ width: `${prediction.probabilities.metaplastic * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label">Parabasal</span>
                              <span className="prob-value">{(prediction.probabilities.parabasal * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar cervical-low" style={{ width: `${prediction.probabilities.parabasal * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label">Superficial-Intermediate</span>
                              <span className="prob-value">{(prediction.probabilities.superficial_intermediate * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar benign" style={{ width: `${prediction.probabilities.superficial_intermediate * 100}%` }} />
                            </div>
                          </div>
                        </>
                      ) : isKidneyCancer ? (
                        <>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label benign-text">Normal</span>
                              <span className="prob-value">{(prediction.probabilities.normal * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar benign" style={{ width: `${prediction.probabilities.normal * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label malignant-text">Tumor</span>
                              <span className="prob-value">{(prediction.probabilities.tumor * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar malignant" style={{ width: `${prediction.probabilities.tumor * 100}%` }} />
                            </div>
                          </div>
                        </>
                      ) : isLungCancer ? (
                        <>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label malignant-text">Adenocarcinoma</span>
                              <span className="prob-value">{(prediction.probabilities.adenocarcinoma * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar malignant" style={{ width: `${prediction.probabilities.adenocarcinoma * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label benign-text">Benign</span>
                              <span className="prob-value">{(prediction.probabilities.benign * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar benign" style={{ width: `${prediction.probabilities.benign * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label malignant-text">Squamous Cell</span>
                              <span className="prob-value">{(prediction.probabilities.squamous_cell * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar malignant" style={{ width: `${prediction.probabilities.squamous_cell * 100}%` }} />
                            </div>
                          </div>
                        </>
                      ) : isColonCancer ? (
                        <>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label malignant-text">Adenocarcinoma</span>
                              <span className="prob-value">{(prediction.probabilities.adenocarcinoma * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar malignant" style={{ width: `${prediction.probabilities.adenocarcinoma * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label benign-text">Benign</span>
                              <span className="prob-value">{(prediction.probabilities.benign * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar benign" style={{ width: `${prediction.probabilities.benign * 100}%` }} />
                            </div>
                          </div>
                        </>
                      ) : isLymphoma ? (
                        <>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label malignant-text">CLL</span>
                              <span className="prob-value">{(prediction.probabilities.cll * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar malignant" style={{ width: `${prediction.probabilities.cll * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label medium-text">Follicular</span>
                              <span className="prob-value">{(prediction.probabilities.fl * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar medium" style={{ width: `${prediction.probabilities.fl * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label malignant-text">MCL</span>
                              <span className="prob-value">{(prediction.probabilities.mcl * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar malignant" style={{ width: `${prediction.probabilities.mcl * 100}%` }} />
                            </div>
                          </div>
                        </>
                      ) : isOralCancer ? (
                        <>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label benign-text">Normal</span>
                              <span className="prob-value">{(prediction.probabilities.normal * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar benign" style={{ width: `${prediction.probabilities.normal * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label malignant-text">SCC</span>
                              <span className="prob-value">{(prediction.probabilities.scc * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar malignant" style={{ width: `${prediction.probabilities.scc * 100}%` }} />
                            </div>
                          </div>
                        </>
                      ) : isBrainCancer ? (
                        <>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label glioma-text">Glioma</span>
                              <span className="prob-value">{(prediction.probabilities.glioma * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar glioma" style={{ width: `${prediction.probabilities.glioma * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label meningioma-text">Meningioma</span>
                              <span className="prob-value">{(prediction.probabilities.meningioma * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar meningioma" style={{ width: `${prediction.probabilities.meningioma * 100}%` }} />
                            </div>
                          </div>
                          <div className="prob-item">
                            <div className="prob-header">
                              <span className="prob-label pituitary-text">Pituitary</span>
                              <span className="prob-value">{(prediction.probabilities.pituitary * 100).toFixed(2)}%</span>
                            </div>
                            <div className="prob-bar-bg">
                              <div className="prob-bar pituitary" style={{ width: `${prediction.probabilities.pituitary * 100}%` }} />
                            </div>
                          </div>
                        </>
                      ) : null}
                    </div>
                  </div>
                )}

                {/* Recommendation */}
                {prediction.prediction.recommendation && (
                  <div className="recommendation-box">
                    <div className="rec-header">
                      <Info size={18} />
                      <h4>Clinical Recommendation</h4>
                    </div>
                    <p>{prediction.prediction.recommendation}</p>
                  </div>
                )}

                {/* Generate Report Button */}
                {!report && (
                  <button className="report-btn" onClick={handleGenerateReport} disabled={reportLoading}>
                    {reportLoading ? (
                      <>
                        <Loader2 size={18} className="spinner" />
                        <span>Generating Report...</span>
                      </>
                    ) : (
                      <>
                        <FileText size={18} />
                        <span>Generate Detailed Report</span>
                      </>
                    )}
                  </button>
                )}

                {/* Full Report */}
                {report && (
                  <div className="full-report">
                    <div className="report-header">
                      <FileText size={20} />
                      <h4>Diagnostic Report</h4>
                      <span className="report-id">ID: {report.report_id?.slice(0, 8)}</span>
                    </div>
                    <div className="report-body">
                      <div className="report-section">
                        <h5>Clinical Information</h5>
                        <p>{report.report?.clinical_information?.description}</p>
                        <div className="report-detail">
                          <strong>Treatment Options:</strong>
                          <span>{report.report?.clinical_information?.typical_treatment_options}</span>
                        </div>
                        <div className="report-detail urgency">
                          <strong>Urgency Level:</strong>
                          <span>{report.report?.clinical_information?.urgency}</span>
                        </div>
                      </div>
                      <div className="report-section">
                        <h5>Recommendations</h5>
                        <div className="rec-list">
                          <div className="rec-group">
                            <h6>Immediate Actions:</h6>
                            <ul>
                              {report.report?.recommendations?.immediate_actions?.slice(0, 3).map((action, i) => (
                                <li key={i}>{action}</li>
                              ))}
                            </ul>
                          </div>
                          <div className="rec-group">
                            <h6>Additional Tests:</h6>
                            <ul>
                              {report.report?.recommendations?.additional_tests?.slice(0, 3).map((test, i) => (
                                <li key={i}>{test}</li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div className="report-disclaimer">
                      <AlertCircle size={14} />
                      <span>{report.disclaimer}</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="disclaimer">
          <AlertCircle size={16} />
          <p><strong>Disclaimer:</strong> This AI tool is for research purposes only. Not a substitute for professional medical diagnosis.</p>
        </div>
      </footer>
    </div>
  )
}

export default App
