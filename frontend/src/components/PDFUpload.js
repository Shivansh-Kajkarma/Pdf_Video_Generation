import React, { useState, useRef } from 'react';
import { uploadPDF } from '../services/api';
import CartesiaConfig from './CartesiaConfig';
import './PDFUpload.css';

const PDFUpload = ({ onUploadSuccess, onBack }) => {
  const [file, setFile] = useState(null);
  const [startPage, setStartPage] = useState(50);
  const [endPage, setEndPage] = useState(50);
  const [voiceProvider, setVoiceProvider] = useState('openai');
  const [openaiVoice, setOpenaiVoice] = useState('');
  const [cartesiaVoiceId, setCartesiaVoiceId] = useState(null);
  const [cartesiaModelId, setCartesiaModelId] = useState('sonic-3');
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileSelect = (selectedFile) => {
    if (selectedFile && selectedFile.type === 'application/pdf') {
      setFile(selectedFile);
      setError(null);
    } else {
      setError('Please select a valid PDF file');
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const droppedFile = e.dataTransfer.files[0];
    handleFileSelect(droppedFile);
  };

  const handleFileInputChange = (e) => {
    const selectedFile = e.target.files[0];
    handleFileSelect(selectedFile);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a PDF file');
      return;
    }

    if (startPage > endPage) {
      setError('Start page must be less than or equal to end page');
      return;
    }

    if (voiceProvider === 'openai' && !openaiVoice) {
      setError('Please select an OpenAI voice');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const response = await uploadPDF(
        file, 
        startPage, 
        endPage, 
        false, 
        voiceProvider,
        voiceProvider === 'openai' ? openaiVoice : null,
        voiceProvider === 'cartesia' ? cartesiaVoiceId : null,
        voiceProvider === 'cartesia' ? cartesiaModelId : null
      );
      onUploadSuccess(response.job_id);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to upload PDF. Please try again.');
      console.error('Upload error:', err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="card">
      <h2>Upload PDF</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Select PDF File</label>
          <div
            className={`file-input-wrapper ${isDragging ? 'dragover' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileInputChange}
              disabled={isUploading}
            />
            <div className="file-input-label">
              {file ? (
                <div>
                  <p>✓ {file.name}</p>
                  <p className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
              ) : (
                <div>
                  <p>📄 Drag and drop your PDF here</p>
                  <p>or click to browse</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="form-row">
          <div className="form-group">
            <label htmlFor="startPage">Start Page</label>
            <input
              id="startPage"
              type="number"
              min="1"
              value={startPage}
              onChange={(e) => setStartPage(parseInt(e.target.value) || 1)}
              disabled={isUploading}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="endPage">End Page</label>
            <input
              id="endPage"
              type="number"
              min="1"
              value={endPage}
              onChange={(e) => setEndPage(parseInt(e.target.value) || 1)}
              disabled={isUploading}
              required
            />
          </div>
        </div>

        <div className="form-group">
          <label htmlFor="voiceProvider">Voice Provider</label>
          <select
            id="voiceProvider"
            value={voiceProvider}
            onChange={(e) => {
              setVoiceProvider(e.target.value);
              if (e.target.value === 'openai') {
                setOpenaiVoice(''); // Reset voice selection when switching to OpenAI
              }
            }}
            disabled={isUploading}
          >
            <option value="openai">OpenAI</option>
            <option value="cartesia">Cartesia</option>
          </select>
        </div>

        {voiceProvider === 'openai' && (
          <div className="form-group">
            <label htmlFor="openaiVoice">OpenAI Voice <span style={{color: 'red'}}>*</span></label>
            <select
              id="openaiVoice"
              value={openaiVoice}
              onChange={(e) => setOpenaiVoice(e.target.value)}
              disabled={isUploading}
              required
            >
              <option value="">-- Select a voice --</option>
              <option value="alloy">Alloy</option>
              <option value="ash">Ash</option>
              <option value="ballad">Ballad</option>
              <option value="coral">Coral</option>
              <option value="echo">Echo</option>
              <option value="fable">Fable</option>
              <option value="nova">Nova</option>
              <option value="onyx">Onyx</option>
              <option value="sage">Sage</option>
              <option value="shimmer">Shimmer</option>
            </select>
          </div>
        )}

        {voiceProvider === 'cartesia' && (
          <CartesiaConfig
            voiceId={cartesiaVoiceId}
            modelId={cartesiaModelId}
            onVoiceChange={setCartesiaVoiceId}
            onModelChange={setCartesiaModelId}
            disabled={isUploading}
          />
        )}

        {error && (
          <div className="alert alert-error">
            {error}
          </div>
        )}

        <div className="button-group">
          {onBack && (
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onBack}
              disabled={isUploading}
            >
              Back
            </button>
          )}
          <button
            type="submit"
            className="btn"
            disabled={isUploading || !file}
            style={onBack ? { flex: 1 } : {}}
          >
            {isUploading ? 'Uploading...' : 'Start Video Generation'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default PDFUpload;

