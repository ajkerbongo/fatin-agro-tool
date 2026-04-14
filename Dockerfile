# Multi-stage build for Fatin Agro Tool (MCPize Wrapper Mode)
FROM node:22-alpine AS builder

WORKDIR /app

# Install all dependencies for build
COPY package*.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# Production stage
FROM node:22-alpine

WORKDIR /app

# Copy package files and install only production dependencies
COPY package*.json ./
RUN npm ci --only=production --ignore-scripts

# Copy built files from builder
COPY --from=builder /app/build ./build

# MCPize / Cloud Run requirements
ENV PORT=8080
EXPOSE 8080

# Health Check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/ping || exit 1

CMD ["node", "build/index.js"]
