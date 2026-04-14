import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { SSEServerTransport } from "@modelcontextprotocol/sdk/server/sse.js";
import {
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
  ListToolsRequestSchema,
  CallToolRequestSchema,
  ErrorCode,
  McpError,
} from "@modelcontextprotocol/sdk/types.js";
import express from "express";
import cors from "cors";
import { z } from "zod";
import * as tf from "@tensorflow/tfjs";
import { Jimp } from "jimp";
import fs from "fs";
import path from "path";

/**
 * Fatin Agro Tool: Global Agriculture Expert
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
  "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___ Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
];

let model: tf.LayersModel | null = null;

async function loadModel() {
  try {
    const fullPath = path.resolve("./model/model.json");
    if (fs.existsSync(fullPath)) {
      model = await tf.loadLayersModel(`file://${fullPath}`);
      console.log("[MCP] Model loaded successfully from local storage.");
    } else {
      console.warn("[MCP] model/model.json not found. The tool will simulate detection until model files are provided.");
    }
  } catch (error) {
    console.error("[MCP] Error loading model:", error);
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
... [Detailed best practices for global farmers] ...
1. Soil Health First.
2. Low-cost organic pest control.
3. Proper water management.
`;

const server = new Server(
  { name: SERVER_NAME, version: SERVER_VERSION },
  { capabilities: { resources: {}, tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "list_supported_crops",
      description: "Lists all supported crops and common diseases.",
      inputSchema: { type: "object", properties: {} },
    },
    {
      name: "detect_crop_disease",
      description: "Analyzes a leaf image (base64) using TensorFlow.js and identifying the crop/disease.",
      inputSchema: {
        type: "object",
        properties: {
          image_base64: { type: "string", description: "Base64 string of leaf image." },
        },
        required: ["image_base64"],
      },
    },
    {
      name: "recommend_treatment",
      description: "Provides practical, organic, and low-cost agricultural advice.",
      inputSchema: {
        type: "object",
        properties: {
          crop: { type: "string" },
          disease: { type: "string" },
          location: { type: "string", optional: true },
        },
        required: ["crop", "disease"],
      },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "list_supported_crops":
      return { content: [{ type: "text", text: JSON.stringify(SUPPORTED_CROPS, null, 2) }] };

    case "detect_crop_disease": {
      const { image_base64 } = z.object({ image_base64: z.string() }).parse(args);
      const buffer = Buffer.from(image_base64, "base64");

      if (!model) {
        return {
          content: [{
            type: "text",
            text: JSON.stringify({
              crop: "Simulator",
              disease: "Healthy Leaf (Simulation)",
              confidence: 0.99,
              note: "Architecture is READY. To enable real detection: 1. Place 'model.json' and '.bin' weights in /model folder. 2. Ensure they match PlantVillage labels."
            }, null, 2)
          }]
        };
      }

      // Process image using Jimp (Pure JS)
      const image = await Jimp.read(buffer);
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

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            crop: crop.replace(/_/g, " "),
            disease: disease.replace(/_/g, " "),
            confidence: scores[maxIndex],
            status: "Real Inference"
          }, null, 2)
        }]
      };
    }

    case "recommend_treatment": {
      const { crop, disease, location } = z.object({
        crop: z.string(),
        disease: z.string(),
        location: z.string().optional()
      }).parse(args);

      let organic = "Apply Neem oil solution (1-2 tsp per liter of water) or use a baking soda solution (1 tsp per liter) for fungal issues.";
      let preventive = "Rotate crops every 3 years. Ensure proper drainage and avoid watering leaves directly.";

      if (disease.toLowerCase().includes("blight")) {
        organic = "Remove and bury/burn infected material immediately. Use copper-based sprays (organic approved) as a last resort.";
      }

      return {
        content: [{
          type: "text",
          text: JSON.stringify({
            recommendation: `Treatment for ${disease} in ${crop} at ${location || 'Global Location'}.`,
            measures: organic,
            prevention: preventive,
            legal_disclaimer: "Always check local agricultural regulations before applying treatments."
          }, null, 2)
        }]
      };
    }

    default:
      throw new McpError(ErrorCode.MethodNotFound, "Tool not found");
  }
});

server.setRequestHandler(ListResourcesRequestSchema, async () => ({
  resources: [{
    uri: "global-agro://knowledge-base",
    name: "Agro Knowledge Base",
    mimeType: "text/markdown"
  }]
}));

server.setRequestHandler(ReadResourceRequestSchema, async (req) => {
  if (req.params.uri === "global-agro://knowledge-base") {
    return {
      contents: [{
        uri: req.params.uri,
        mimeType: "text/markdown",
        text: KNOWLEDGE_BASE_CONTENT
      }]
    };
  }
  throw new McpError(ErrorCode.InvalidRequest, "Resource not found");
});

const app = express();
app.use(cors());
app.use(express.json());

let transport: SSEServerTransport | null = null;

app.get("/sse", async (req, res) => {
  transport = new SSEServerTransport("/messages", res);
  await server.connect(transport);
});

app.post("/messages", async (req, res) => {
  if (!transport) return res.status(400).send("No channel");
  await transport.handlePostMessage(req, res);
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, async () => {
  console.log(`[Agro] Running on http://localhost:${PORT}/sse`);
  await loadModel();
});
