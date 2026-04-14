# Fatin Agro Tool MCP Server

A complete, production-ready Model Context Protocol (MCP) server designed to help farmers worldwide. This tool provides crop disease detection (AI-ready), treatment recommendations, and a global agricultural knowledge base.

## Features

- **detect_crop_disease**: Analyzes leaf images for common diseases (supports major global crops).
- **recommend_treatment**: Provides organic-first, low-cost, step-by-step agricultural advice.
- **list_supported_crops**: Reference guide for farmers and AI agents on supported crops.
- **Global Agriculture Resource**: A knowledge base with best practices and region-specific tips.

## Prerequisites

- Node.js (v18+)
- npm

## Local Development

1. Install dependencies:
   ```bash
   npm install
   ```
2. Build the project:
   ```bash
   npm run build
   ```
3. Start the server:
   ```bash
   npm start
   ```
   The server will start an SSE-based HTTP server on port 3000.

### Hosting on Render (FREE)

This app is ready for "one-click" deployment on **Render.com** using the included `render.yaml` Blueprint.

1.  **Push to GitHub**: Upload this project to your GitHub account.
2.  **Render Dashboard**: Log in to [dashboard.render.com](https://dashboard.render.com).
3.  **New Blueprint**: Click **"New"** -> **"Blueprint"** and connect your GitHub repository.
4.  **Automatic Setup**: Render will automatically detect the settings and deploy your global agricultural advisor.

*Your server will be live at:* `https://your-app-name.onrender.com/sse`

---

### Hosting on MCPize
This server also supports direct deployment on **[MCPize](https://mcpize.com)**.

*   **Build Command**: `npm install && npm run build`
*   **Start Command**: `npm start`
*   **Port**: `3000`
*   **Protocol**: `SSE`

## Real-Time Detection: TensorFlow.js
The server is now fully equipped with a **real** TensorFlow.js engine. To enable live inference:

1.  **Prepare Model**: Export your Keras/TensorFlow model to the **TensorFlow.js Layers format** (using `tensorflowjs_converter`).
2.  **Add Files**: Create a folder named `model` in the root directory.
3.  **Place Files**: Put your `model.json` and `.bin` weight files inside that folder.
4.  **Restart**: The server will automatically detect and load the model on startup.

If no model is present, the tool runs in **Simulation Mode** (returning a mock healthy result) so the MCP connection stays healthy.

## Build and Run

```bash
npm install
npm run build
npm start
```

## License
MIT
