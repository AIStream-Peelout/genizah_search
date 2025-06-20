# src/frontend/Dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Accept build argument for API URL
ARG REACT_APP_API_URL=http://localhost:8000
ENV REACT_APP_API_URL=$REACT_APP_API_URL

# Debug: Show what API URL is being used
RUN echo "Building with API URL: $REACT_APP_API_URL"

# Build the app
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app
COPY --from=builder /app/build /usr/share/nginx/html

# Copy custom nginx config if it exists
COPY nginx.conf /etc/nginx/conf.d/default.conf 2>/dev/null || echo "No custom nginx config found, using default"

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]