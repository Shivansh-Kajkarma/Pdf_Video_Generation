import React, { useState, useEffect, useCallback } from 'react';
import { listCartesiaVoices, listCartesiaModels, getCartesiaVoice } from '../services/api';
import './CartesiaConfig.css';

const CartesiaConfig = ({ voiceId, modelId, onVoiceChange, onModelChange, disabled = false }) => {
  const [voices, setVoices] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(voiceId || '');
  const [selectedModel, setSelectedModel] = useState(modelId || 'sonic-3');
  const [voiceDetails, setVoiceDetails] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [voiceInputMode, setVoiceInputMode] = useState('dropdown'); // 'dropdown' or 'manual'
  const [manualVoiceId, setManualVoiceId] = useState('');
  const [isUserControlled, setIsUserControlled] = useState(false); // Track if user manually changed mode

  const loadVoiceDetails = useCallback(async (voiceIdToLoad) => {
    try {
      const details = await getCartesiaVoice(voiceIdToLoad);
      setVoiceDetails(details);
    } catch (err) {
      console.error('Error loading voice details:', err);
      // Find voice in local list as fallback
      const voice = voices.find(v => v.id === voiceIdToLoad);
      if (voice) {
        setVoiceDetails(voice);
      }
    }
  }, [voices]);

  const loadCartesiaData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [voicesResponse, modelsResponse] = await Promise.all([
        listCartesiaVoices('en'), // Filter for English voices by default
        listCartesiaModels()
      ]);
      const loadedVoices = voicesResponse.voices || [];
      setVoices(loadedVoices);
      setModels(modelsResponse.models || []);
      
      // Only set initial mode if user hasn't manually controlled it
      if (!isUserControlled) {
        // Check if the initial voiceId is in the dropdown list
        if (voiceId) {
          const voiceInList = loadedVoices.find(v => v.id === voiceId);
          if (voiceInList) {
            // Voice is in the list, use dropdown mode
            setSelectedVoice(voiceId);
            setVoiceInputMode('dropdown');
          } else {
            // Voice is not in the list, use manual mode
            setManualVoiceId(voiceId);
            setVoiceInputMode('manual');
          }
        } else if (loadedVoices.length > 0) {
          // Set default voice if none selected
          const defaultVoice = loadedVoices.find(v => v.id === '98a34ef2-2140-4c28-9c71-663dc4dd7022') 
            || loadedVoices[0];
          setSelectedVoice(defaultVoice.id);
          setVoiceInputMode('dropdown');
        }
      }
    } catch (err) {
      console.error('Error loading Cartesia data:', err);
      setError('Failed to load Cartesia voices and models. Using defaults.');
      // Set fallback defaults
      const fallbackVoices = [
        {
          id: '98a34ef2-2140-4c28-9c71-663dc4dd7022',
          name: 'Tessa',
          language: 'en',
          tags: ['Emotive', 'Expressive'],
          description: 'Expressive American English voice'
        }
      ];
      setVoices(fallbackVoices);
      setModels([
        { id: 'sonic-3', name: 'Sonic 3', description: 'Latest streaming TTS model' }
      ]);
      
      // Only set initial mode if user hasn't manually controlled it
      if (!isUserControlled) {
        // Check if voiceId is in fallback list
        if (voiceId) {
          const voiceInList = fallbackVoices.find(v => v.id === voiceId);
          if (voiceInList) {
            setSelectedVoice(voiceId);
            setVoiceInputMode('dropdown');
          } else {
            setManualVoiceId(voiceId);
            setVoiceInputMode('manual');
          }
        } else {
          setSelectedVoice(fallbackVoices[0].id);
          setVoiceInputMode('dropdown');
        }
      }
    } finally {
      setLoading(false);
    }
  }, [voiceId, isUserControlled]);

  useEffect(() => {
    loadCartesiaData();
  }, [loadCartesiaData]);

  // Handle external voiceId prop changes (only when not user-controlled)
  useEffect(() => {
    // Only auto-switch mode if user hasn't manually controlled it
    if (!isUserControlled && voiceId && voiceId !== selectedVoice && voiceId !== manualVoiceId) {
      // Check if voiceId is in the loaded voices list
      const voiceInList = voices.find(v => v.id === voiceId);
      if (voiceInList) {
        setSelectedVoice(voiceId);
        setVoiceInputMode('dropdown');
        setManualVoiceId('');
      } else {
        setManualVoiceId(voiceId);
        setVoiceInputMode('manual');
        setSelectedVoice('');
      }
    }
  }, [voiceId, voices, selectedVoice, manualVoiceId, isUserControlled]);

  useEffect(() => {
    const voiceIdToLoad = voiceInputMode === 'manual' ? manualVoiceId : selectedVoice;
    if (voiceIdToLoad) {
      loadVoiceDetails(voiceIdToLoad);
    } else {
      setVoiceDetails(null);
    }
  }, [selectedVoice, manualVoiceId, voiceInputMode, loadVoiceDetails]);

  useEffect(() => {
    if (onVoiceChange) {
      // Use manual voice ID if in manual mode, otherwise use selected voice from dropdown
      const voiceToUse = voiceInputMode === 'manual' ? manualVoiceId : selectedVoice;
      onVoiceChange(voiceToUse);
    }
  }, [selectedVoice, manualVoiceId, voiceInputMode, onVoiceChange]);

  useEffect(() => {
    if (onModelChange) {
      onModelChange(selectedModel);
    }
  }, [selectedModel, onModelChange]);


  const handleVoiceChange = (e) => {
    setSelectedVoice(e.target.value);
  };

  const handleModelChange = (e) => {
    setSelectedModel(e.target.value);
  };

  const handleVoiceInputModeChange = (mode) => {
    setIsUserControlled(true); // Mark as user-controlled
    setVoiceInputMode(mode);
    if (mode === 'manual') {
      // Clear dropdown selection when switching to manual
      setSelectedVoice('');
    } else {
      // Clear manual input when switching to dropdown
      setManualVoiceId('');
    }
  };

  const handleManualVoiceIdChange = (e) => {
    const value = e.target.value;
    setManualVoiceId(value);
  };

  if (loading) {
    return (
      <div className="cartesia-config">
        <div className="loading">Loading Cartesia configuration...</div>
      </div>
    );
  }

  return (
    <div className="cartesia-config">
      {error && (
        <div className="alert alert-warning">
          {error}
        </div>
      )}

      <div className="form-group">
        <label htmlFor="cartesiaModel">Model</label>
        <select
          id="cartesiaModel"
          value={selectedModel}
          onChange={handleModelChange}
          disabled={disabled}
        >
          {models.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name || model.id}
            </option>
          ))}
        </select>
        {models.find(m => m.id === selectedModel)?.description && (
          <small className="form-help">
            {models.find(m => m.id === selectedModel).description}
          </small>
        )}
      </div>

      <div className="form-group">
        <label htmlFor="cartesiaVoice">Voice</label>
        
        {/* Voice Input Mode Toggle */}
        <div className="voice-input-mode-toggle">
          <div className="toggle-option">
            <input
              type="radio"
              name="voiceInputMode"
              value="dropdown"
              checked={voiceInputMode === 'dropdown'}
              onChange={() => handleVoiceInputModeChange('dropdown')}
              disabled={disabled}
              id="voiceModeDropdown"
            />
            <label htmlFor="voiceModeDropdown">Select from list</label>
          </div>
          <div className="toggle-option">
            <input
              type="radio"
              name="voiceInputMode"
              value="manual"
              checked={voiceInputMode === 'manual'}
              onChange={() => handleVoiceInputModeChange('manual')}
              disabled={disabled}
              id="voiceModeManual"
            />
            <label htmlFor="voiceModeManual">Enter Voice ID</label>
          </div>
        </div>

        {/* Dropdown Mode */}
        {voiceInputMode === 'dropdown' && (
          <select
            id="cartesiaVoice"
            value={selectedVoice}
            onChange={handleVoiceChange}
            disabled={disabled}
          >
            <option value="">Select a voice...</option>
            {voices.map((voice) => (
              <option key={voice.id} value={voice.id}>
                {voice.name || voice.id} {voice.tags && voice.tags.length > 0 && `(${voice.tags.join(', ')})`}
              </option>
            ))}
          </select>
        )}

        {/* Manual Input Mode */}
        {voiceInputMode === 'manual' && (
          <input
            type="text"
            id="cartesiaVoiceManual"
            value={manualVoiceId}
            onChange={handleManualVoiceIdChange}
            placeholder="Enter Cartesia Voice ID (e.g., 98a34ef2-2140-4c28-9c71-663dc4dd7022)"
            disabled={disabled}
            className="voice-id-input"
          />
        )}

        {/* Voice Details Display */}
        {voiceDetails && (voiceInputMode === 'dropdown' ? selectedVoice : manualVoiceId) && (
          <div className="voice-details">
            {voiceDetails.description && (
              <p className="voice-description">{voiceDetails.description}</p>
            )}
            {voiceDetails.tags && voiceDetails.tags.length > 0 && (
              <div className="voice-tags">
                {voiceDetails.tags.map((tag, idx) => (
                  <span key={idx} className="tag">{tag}</span>
                ))}
              </div>
            )}
            {voiceDetails.language && (
              <small className="voice-language">Language: {voiceDetails.language}</small>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CartesiaConfig;

