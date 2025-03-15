/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        'political-low': '#34d399', // Green for neutral 
        'political-high': '#ef4444', // Red for strong bias
        'rhetoric-low': '#3b82f6', // Blue for informative
        'rhetoric-high': '#ef4444', // Red for emotional
        'depth-low': '#cbd5e1', // Light gray for overview
        'depth-medium': '#94a3b8', // Medium gray for analysis
        'depth-high': '#475569', // Dark gray for in-depth
      },
    },
  },
  plugins: [],
}