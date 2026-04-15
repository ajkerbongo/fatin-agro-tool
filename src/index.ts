import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express, { Request, Response } from "express";
import cors from "cors";
import { z } from "zod";
import chalk from "chalk";
import * as tf from "@tensorflow/tfjs";
import { Jimp } from "jimp";
import fs from "fs";
import path from "path";

/**
 * Fatin Agro Tool: Global Agriculture Expert (MCPize Edition)
 */

const SERVER_NAME = "Fatin Agro Tool";
const SERVER_VERSION = "1.0.0";

// PlantVillage Labels mapping
const LABELS = [
  "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
  "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy",
  "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust_", "Corn___Northern_Leaf_Blight", "Corn___healthy",
  "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
  "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
  "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
  "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
  "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
  "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
];

let model: tf.LayersModel | null = null;

async function loadModel() {
  try {
    const modelPath = process.env.MODEL_PATH || path.join(process.cwd(), "model", "model.json");
    if (fs.existsSync(modelPath)) {
      model = await tf.loadLayersModel(`file://${modelPath}`);
      console.log(chalk.green("[MCP] Model loaded successfully from local storage."));
    } else {
      console.warn(chalk.yellow("[MCP] model/model.json not found. The tool will simulate detection until model files are provided."));
    }
  } catch (error) {
    console.error(chalk.red("[MCP] Error loading model:"), error);
  }
}

const SUPPORTED_CROPS = {
  "PlantVillage Standards": [
    "Apple", "Blueberry", "Cherry", "Corn", "Grape", "Orange", "Peach", "Bell Pepper", "Potato", "Raspberry", "Soybean", "Squash", "Strawberry", "Tomato"
  ],
  "Global/Tropical Crops": [
    "Rice", "Wheat", "Jute", "Maize", "Banana", "Cassava", "Sugarcane", "Mango", "Coffee", "Cotton"
  ]
};

const KNOWLEDGE_BASE_CONTENT = `
# Global Agriculture Knowledge Base
1. Soil Health First.
2. Low-cost organic pest control.
3. Proper water management.
`;

// ============================================================================
// Dev Logging Utilities
// ============================================================================

const isDev = process.env.NODE_ENV !== "production";

function timestamp(): string {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}

function logRequest(method: string, params?: unknown): void {
  if (!isDev) return;
  const paramsStr = params ? chalk.gray(` ${JSON.stringify(params).slice(0, 100)}`) : "";
  console.log(`${chalk.gray(`[${timestamp()}]`)} ${chalk.cyan("→")} ${method}${paramsStr}`);
}

// ============================================================================
// MCP Server Setup
// ============================================================================

const server = new McpServer({
  name: SERVER_NAME,
  version: SERVER_VERSION,
});

// Resources
server.registerResource(
  "knowledge-base",
  "global-agro://knowledge-base",
  {
    description: "Detailed best practices for global farmers",
    mimeType: "text/markdown",
  },
  async () => ({
    contents: [{
      uri: "global-agro://knowledge-base",
      text: KNOWLEDGE_BASE_CONTENT,
    }],
  })
);

// Tools
server.registerTool(
  "list_supported_crops",
  {
    description: "Lists all supported crops and common diseases.",
    inputSchema: z.object({}),
  },
  async () => ({
    content: [{ type: "text", text: JSON.stringify(SUPPORTED_CROPS, null, 2) }],
    structuredContent: SUPPORTED_CROPS
  })
);

server.registerTool(
  "detect_crop_disease",
  {
    description: "Analyzes a leaf image (base64) using TensorFlow.js and identifying the crop/disease.",
    inputSchema: z.object({
      image_base64: z.string().describe("Base64 string of leaf image."),
    }),
  },
  async ({ image_base64 }) => {
    const buffer = Buffer.from(image_base64, "base64");
    if (!model) {
      const simulatorResult = {
        crop: "Simulator",
        disease: "Healthy Leaf (Simulation)",
        confidence: 0.99,
        note: "Real Inference ready. Place model files in /model folder."
      };

      return {
        content: [{
          type: "text",
          text: JSON.stringify(simulatorResult, null, 2)
        }],
        structuredContent: simulatorResult
      };
    }

    const image = await Jimp.read(buffer as any);
    image.cover({ w: 224, h: 224 });
    const floatData = new Float32Array(224 * 224 * 3);
    image.scan(0, 0, 224, 224, (x, y, idx) => {
      const pixelIdx = (y * 224 + x) * 3;
      floatData[pixelIdx] = image.bitmap.data[idx] / 255;
      floatData[pixelIdx + 1] = image.bitmap.data[idx + 1] / 255;
      floatData[pixelIdx + 2] = image.bitmap.data[idx + 2] / 255;
    });

    const tensor = tf.tensor3d(floatData, [224, 224, 3]).expandDims();
    const prediction = model.predict(tensor) as tf.Tensor;
    const scores = await prediction.data();
    const maxIndex = scores.indexOf(Math.max(...scores));
    const label = LABELS[maxIndex] || "Unknown Crop/Disease";
    const [crop, disease] = label.includes("___") ? label.split("___") : [label, "Unknown"];

    // Clean up tensors to prevent memory leaks
    tf.dispose([tensor, prediction]);

    const result = {
      crop: crop.replace(/_/g, " "),
      disease: disease.replace(/_/g, " "),
      confidence: scores[maxIndex],
      status: "Real Inference"
    };

    return {
      content: [{
        type: "text",
        text: JSON.stringify(result, null, 2)
      }],
      structuredContent: result
    };
  }
);

server.registerTool(
  "recommend_treatment",
  {
    description: "Provides practical, organic, and low-cost agricultural advice.",
    inputSchema: z.object({
      crop: z.string().describe("Name of the crop (e.g. Tomato, Potato, Apple)"),
      disease: z.string().describe("Detected disease name"),
      location: z.string().optional().describe("Optional geographic location for region-specific advice"),
    }),
  },
  async ({ crop, disease, location }) => {
    let organic = "Apply Neem oil solution (1-2 tsp per liter of water) or use a baking soda solution (1 tsp per liter) for fungal issues.";
    let preventive = "Rotate crops every 3 years. Ensure proper drainage and avoid watering leaves directly.";

    if (disease.toLowerCase().includes("blight")) {
      organic = "Remove and bury/burn infected material immediately. Use copper-based sprays (organic approved) as a last resort.";
    }

    const recommendationResult = {
      recommendation: `Treatment for ${disease} in ${crop} at ${location || 'Global Location'}.`,
      measures: organic,
      prevention: preventive,
    };

    return {
      content: [{
        type: "text",
        text: JSON.stringify(recommendationResult, null, 2)
      }],
      structuredContent: recommendationResult
    };
  }
);

// ============================================================================
// Express App Setup
// ============================================================================

const app = express();
app.use(cors());
app.use(express.json());

// Landing Page for Visual Feedback
app.get("/", (req, res) => {
  res.send(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>${SERVER_NAME} - MCP Server</title>
      <style>
        body { font-family: sans-serif; background: #0f172a; color: #f8fafc; display: flex; align-items: center; justify-content: center; min-height: 100vh; margin: 0; }
        .card { background: rgba(30, 41, 59, 0.7); padding: 3rem; border-radius: 2rem; text-align: center; border: 1px solid rgba(255, 255, 255, 0.1); }
        h1 { color: #4ade80; }
        .status { color: #4ade80; font-weight: bold; margin-bottom: 1rem; }
      </style>
    </head>
    <body>
      <div class="card">
        <div class="status">● MCP SERVER ONLINE</div>
        <h1>${SERVER_NAME}</h1>
        <p>Global Agricultural Expertise & Disease Detection</p>
        <p>Endpoint: <code>/mcp</code></p>
      </div>
    </body>
    </html>
  `);
});

app.get("/health", (_req, res) => {
  res.status(200).json({ status: "healthy" });
});

app.get("/ping", (_req, res) => {
  res.status(200).send("pong");
});

const transport = new StreamableHTTPServerTransport({
  sessionIdGenerator: undefined,
  enableJsonResponse: true,
});

server.connect(transport);

app.post("/mcp", async (req: Request, res: Response) => {
  try {
    logRequest(req.body?.method || "unknown", req.body?.params);
    await transport.handleRequest(req, res, req.body);
  } catch (error) {
    console.error("[MCP ERROR]", error);
    res.status(500).json({ error: "Handler failed", message: (error as Error).message });
  }
});

// ============================================================================
// Start Server
// ============================================================================

const port = parseInt(process.env.PORT || "8080");
const httpServer = app.listen(port, "0.0.0.0", async () => {
  console.log(chalk.bold("\n[Agro] Server running on"), chalk.cyan(`http://localhost:${port}`));
  console.log(`  ${chalk.gray("Global Agro Knowledge Base initialized.")}`);
  await loadModel();
});

process.on("SIGTERM", () => {
  httpServer.close(() => process.exit(0));
});
