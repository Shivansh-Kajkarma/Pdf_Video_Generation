import React, { useState } from 'react';
import { generateReelsVideo } from '../services/api';
import CartesiaConfig from './CartesiaConfig';
import './ReelsShorts.css';

const ReelsShorts = ({ onVideoGenerated, onBack }) => {
  const [text, setText] = useState('');
  const [voiceProvider, setVoiceProvider] = useState('openai');
  const [openaiVoice, setOpenaiVoice] = useState('');
  const [cartesiaVoiceId, setCartesiaVoiceId] = useState(null);
  const [cartesiaModelId, setCartesiaModelId] = useState('sonic-3');
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [videoJobId, setVideoJobId] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!text || !text.trim()) {
      setError('Please enter text for the video');
      return;
    }

    if (voiceProvider === 'openai' && !openaiVoice) {
      setError('Please select an OpenAI voice');
      return;
    }

    setIsGenerating(true);
    setError(null);

    try {
      const response = await generateReelsVideo(
        text,
        voiceProvider,
        voiceProvider === 'openai' ? openaiVoice : null,
        voiceProvider === 'cartesia' ? cartesiaVoiceId : null,
        voiceProvider === 'cartesia' ? cartesiaModelId : null
      );
      
      setVideoJobId(response.job_id);
      onVideoGenerated(response);
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start video generation. Please try again.');
      console.error('Video generation error:', err);
      setIsGenerating(false);
    }
  };

  const wordCount = text.trim().split(/\s+/).filter(word => word.length > 0).length;
  const charCount = text.trim().length;

  return (
    <div className="card reels-shorts">
      <div className="card-header">
        <h2>Create Reels/Shorts Video</h2>
        <p className="subtitle">Enter your text to create a social media video</p>
      </div>

      {!videoJobId ? (
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>
              Enter Your Text
              <span className="text-count">({wordCount.toLocaleString()} words, {charCount.toLocaleString()} characters)</span>
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={12}
              className="reels-textarea"
              placeholder="Enter your text here... This will be used to generate a video with narration."
              disabled={isGenerating}
            />
            <p className="help-text">
              The text will be converted to speech and displayed on screen as the video plays.
            </p>
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
              disabled={isGenerating}
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
                disabled={isGenerating}
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
              disabled={isGenerating}
            />
          )}

          {error && (
            <div className="alert alert-error">
              {error}
            </div>
          )}

          <div className="button-group">
            <button
              type="button"
              className="btn btn-secondary"
              onClick={onBack}
              disabled={isGenerating}
            >
              Back
            </button>
            <button
              type="submit"
              className="btn"
              disabled={isGenerating || !text.trim()}
            >
              {isGenerating ? 'Generating Video...' : 'Generate Video'}
            </button>
          </div>
        </form>
      ) : (
        <div className="status-container">
          <div className="status-info">
            <h3>Video Generation Started</h3>
            <p className="status-message">
              Your reels/shorts video is being generated.
            </p>
            <p className="status-message">
              Job ID: {videoJobId}
            </p>
            <p className="status-message">
              Redirecting to track progress...
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ReelsShorts;

