import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout for most requests
});

// Add request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for debugging
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`, response.data);
    return response;
  },
  (error) => {
    console.error('API Response Error:', {
      message: error.message,
      status: error.response?.status,
      data: error.response?.data,
      url: error.config?.url
    });
    return Promise.reject(error);
  }
);

export const uploadPDF = async (file, startPage, endPage, generateSummary = false, voiceProvider = 'openai', openaiVoice = null, cartesiaVoiceId = null, cartesiaModelId = null) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('start_page', startPage);
  formData.append('end_page', endPage);
  formData.append('generate_summary', generateSummary);
  formData.append('voice_provider', voiceProvider);
  if (voiceProvider === 'openai' && openaiVoice) {
    formData.append('openai_voice', openaiVoice);
  }
  if (cartesiaVoiceId) {
    formData.append('cartesia_voice_id', cartesiaVoiceId);
  }
  if (cartesiaModelId) {
    formData.append('cartesia_model_id', cartesiaModelId);
  }

  const response = await api.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 60000, // 60 seconds for file upload
  });
  return response.data;
};

export const getJobStatus = async (jobId) => {
  const response = await api.get(`/api/jobs/${jobId}`, {
    timeout: 5000, // 5 seconds for status checks (should be fast)
  });
  return response.data;
};

export const downloadVideo = async (jobId) => {
  const response = await api.get(`/api/jobs/${jobId}/download/video`, {
    responseType: 'blob',
  });
  return response.data;
};

export const downloadSummary = async (jobId) => {
  const response = await api.get(`/api/jobs/${jobId}/download/summary`, {
    responseType: 'text',
  });
  return response.data;
};

export const generateSummaryVideo = async (jobId) => {
  const response = await api.post(`/api/jobs/${jobId}/generate-summary-video`);
  return response.data;
};

export const downloadSummaryVideo = async (jobId) => {
  const response = await api.get(`/api/jobs/${jobId}/download/summary-video`, {
    responseType: 'blob',
  });
  return response.data;
};

export const generateSummary = async (jobId) => {
  const response = await api.post(`/api/jobs/${jobId}/generate-summary`);
  return response.data;
};

export const summarizePDF = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/summarize-pdf', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 300000, // 5 minutes for summarization
  });
  return response.data;
};

export const generateVideoFromText = async (text, voiceProvider = 'openai', openaiVoice = null, cartesiaVoiceId = null, cartesiaModelId = null) => {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('voice_provider', voiceProvider);
  if (voiceProvider === 'openai' && openaiVoice) {
    formData.append('openai_voice', openaiVoice);
  }
  if (cartesiaVoiceId) {
    formData.append('cartesia_voice_id', cartesiaVoiceId);
  }
  if (cartesiaModelId) {
    formData.append('cartesia_model_id', cartesiaModelId);
  }

  const response = await api.post('/api/generate-video-from-text', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 600000, // 10 minutes for video generation
  });
  return response.data;
};

export const generateReelsVideo = async (text, voiceProvider = 'openai', openaiVoice = null, cartesiaVoiceId = null, cartesiaModelId = null) => {
  const formData = new FormData();
  formData.append('text', text);
  formData.append('voice_provider', voiceProvider);
  if (voiceProvider === 'openai' && openaiVoice) {
    formData.append('openai_voice', openaiVoice);
  }
  if (cartesiaVoiceId) {
    formData.append('cartesia_voice_id', cartesiaVoiceId);
  }
  if (cartesiaModelId) {
    formData.append('cartesia_model_id', cartesiaModelId);
  }

  const response = await api.post('/api/generate-reels-video', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
    timeout: 600000, // 10 minutes for video generation
  });
  return response.data;
};

// Cartesia API functions
export const listCartesiaVoices = async (language = null, tags = null) => {
  const params = {};
  if (language) params.language = language;
  if (tags) params.tags = tags;
  const response = await api.get('/api/cartesia/voices', { params });
  return response.data;
};

export const getCartesiaVoice = async (voiceId) => {
  const response = await api.get(`/api/cartesia/voices/${voiceId}`);
  return response.data;
};

export const listCartesiaModels = async () => {
  const response = await api.get('/api/cartesia/models');
  return response.data;
};

export default api;

