// Global application configuration
const config = {
  // Backend API URL - prioritize environment variable or fallback to default
  API_URL: process.env.REACT_APP_API_URL || 'http://localhost:8000',
  
  // Other app configuration can be added here
  APP_NAME: 'MINDSET News Analytics',
  DEFAULT_TIMEOUT: 15000, // 15 seconds
};

export default config;