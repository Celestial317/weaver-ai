# Weaver AI

## Project Structure
```
celestial317-weaver-ai/
├── README.md
├── class2idx.json               # Color class mappings
├── load_models.py               # Model initialization script
├── model_for_gen.py             # Model generation utilities
├── model_weights.pth            # Trained ResNet model
├── subclass2idx.json            # Color subclass mappings
├── vector_db_resnet.py          # ResNet-based vector database setup
├── react_app/                   # React frontend application
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx
│       ├── index.css
│       ├── main.tsx
│       ├── vite-env.d.ts
│       ├── components/
│       │   ├── FileUpload.tsx
│       │   ├── Footer.tsx
│       │   ├── LoadingSpinner.tsx
│       │   └── Navbar.tsx
│       └── pages/
│           ├── AISearch.tsx
│           ├── Catalogue.tsx
│           ├── HomePage.tsx
│           ├── ImageRecommendations.tsx
│           ├── Stylizer.tsx
│           ├── VirtualTryOn.tsx
│           └── VisualDesigner.tsx
└── server_scripts/              # Backend server scripts
    ├── ai_search.py             # AI search server
    ├── amalgam.py               # Amalgam server
    ├── designer.py              # Designer server
    └── recommender.py           # Recommendation server
```

## How to Run the Full Web Application

### Prerequisites

Before starting, make sure you have the following environment variables configured:
- `GEMINI_API_KEY` - Your Google Gemini API key
- `QDRANT_API_KEY` - Your Qdrant API key
- `QDRANT_URL` - Your Qdrant database URL

### Step 1: Initial Setup and Dependencies

1. **Navigate to the project directory:**
   ```bash
   cd weaver-ai
   ```

2. **Load and initialize models:**
   ```bash
   python load_models.py
   python model_for_gen.py
   ```

### Step 2: Setup Vector Database

1. **Populate the vector database:**
   ```bash
   python vector_db_resnet.py
   ```
   
   This script will:
   - Initialize the Qdrant vector database
   - Process and embed fashion data using ResNet
   - Create searchable vector collections

### Step 3: Start Backend Servers

1. **Start all backend servers (run each in a separate terminal):**
   
   **Terminal 1 - AI Search Server:**
   ```bash
   python server_scripts/ai_search.py
   ```
   
   **Terminal 2 - Recommendation Server:**
   ```bash
   python server_scripts/recommender.py
   ```
   
   **Terminal 3 - Designer Server:**
   ```bash
   python server_scripts/designer.py
   ```
   
   **Terminal 4 - Amalgam Server:**
   ```bash
   python server_scripts/amalgam.py
   ```

### Step 4: Setup React Frontend

1. **Navigate to the React project:**
   ```bash
   cd react_app
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Start the React development server:**
   ```bash
   npm run dev
   ```

4. **Open the web application:**
   - The React app will be available at: http://localhost:5173
   - Navigate through the different pages using the navigation menu

### Step 5: Using the Application

The application now features multiple AI-powered pages:

1. **Home Page (/):**
   - Landing page with navigation to all features
   - Overview of available AI tools

2. **Image Recommendations (/image-recommendations):**
   - Upload an image for AI-powered clothing recommendations
   - Color palette analysis using ResNet
   - Matching clothing items with confidence scores

3. **AI Search (/ai-search):**
   - Text-based fashion search powered by AI
   - Natural language queries for clothing items
   - Semantic search through the fashion database

4. **Virtual Try-On (/virtual-tryon):**
   - Virtual clothing try-on capabilities
   - AI-powered garment fitting simulation

5. **Stylizer (/stylizer):**
   - AI-powered styling suggestions
   - Outfit coordination and recommendations
   - Generates style transfer of two cloth images

6. **Visual Designer (/visual-designer):**
   - AI-assisted fashion design tools
   - Creative design generation capabilities
   - Ingest drawing, will return cloth

7. **Catalogue (/catalogue):**
   - Browse the complete fashion database
   - Filter and search through available items

## API Endpoints

### Backend Servers

The application runs multiple specialized servers:

#### AI Search Server
- **Endpoint**: `/ai-search`
- **Features**: Natural language fashion search
- **AI Model**: semantic search

#### Recommendation Server  
- **Endpoint**: `/recommend`
- **Features**: Image-based clothing recommendations
- **AI Model**: ResNet color analysis + vector similarity

#### Designer Server
- **Endpoint**: `/designer`
- **Features**: AI-powered design generation
- **AI Model**: Generative AI for fashion design

#### Amalgam Server
- **Endpoint**: `/amalgam`
- **Features**: Combined AI services orchestrator
- **AI Model**: Multi-modal AI coordination

### API Configuration

Make sure to configure the following in your environment:

```bash
# Gemini AI Configuration
export GEMINI_API_KEY="your_gemini_api_key_here"

# Qdrant Vector Database Configuration  
export QDRANT_API_KEY="your_qdrant_api_key_here"
export QDRANT_URL="your_qdrant_cluster_url_here"
```

## Features

### Multi-Modal AI Platform
- **ResNet Analysis**: Deep color and pattern recognition
- **Gemini AI**: Natural language processing and generation
- **Vector Database**: Qdrant-powered similarity search
- **IP Adapter with ControlNet** : for amalgam generation

### Advanced AI Features
- **Semantic Search**: Natural language fashion queries
- **Visual Recognition**: Image-based style analysis
- **Style Generation**: AI-powered design creation
- **Virtual Try-On**: Realistic garment visualization
- **Smart Recommendations**: Multi-factor matching algorithms

### User Experience
- **Modern React Interface**: Responsive, mobile-friendly design
- **Real-time Processing**: Fast AI inference and responses
- **Multi-Page Navigation**: Dedicated features for each AI capability
- **Drag & Drop Upload**: Intuitive file handling
- **Live Feedback**: Progress indicators and status updates

### Technical Architecture
- **Microservices**: Specialized servers for each AI feature
- **Vector Storage**: Efficient similarity search with Qdrant
- **Model Management**: Centralized AI model loading and caching
- **API Integration**: FastAPI services with CORS support
- **Development Tools**: Hot reload, TypeScript, Tailwind CSS

## Troubleshooting

### Common Issues

1. **API Keys Not Set:**
   ```bash
   # Set environment variables
   export GEMINI_API_KEY="your_api_key"
   export QDRANT_API_KEY="your_api_key"
   export QDRANT_URL="your_cluster_url"
   ```

2. **Vector Database Not Populated:**
   ```bash
   # Re-run the database setup
   python vector_db_resnet.py
   ```

3. **Model Loading Issues:**
   ```bash
   # Reload models
   python load_models.py
   ```

4. **Server Port Conflicts:**
   ```bash
   # Check for running processes
   netstat -ano | findstr :8000
   # Kill conflicting processes
   taskkill /PID <process_id> /F
   ```

5. **React Build Issues:**
   ```bash
   cd react_app
   rm -rf node_modules package-lock.json
   npm install
   ```

### Development Tips

1. **Testing Individual Servers:**
   ```bash
   # Test each server endpoint individually
   curl http://localhost:8000/health
   curl http://localhost:8001/status
   ```

2. **Monitoring Logs:**
   - Each server outputs logs to its terminal
   - Check browser console for frontend errors
   - Monitor Qdrant dashboard for database status

3. **Model Performance:**
   - Ensure sufficient GPU/CPU resources
   - Monitor memory usage during inference
   - Check model loading completion before testing

## Success Indicators

**Models Ready:**
- Model initialization completed successfully
- ResNet model weights loaded
- Vector database populated with embeddings

**Servers Running:**
- AI Search server: Running and responsive
- Recommendation server: Image processing functional
- Designer server: Generation capabilities active
- Amalgam server: Service orchestration working

**Frontend Ready:**
- React app running on http://localhost:5173
- All pages accessible via navigation
- Component rendering without errors
- API calls connecting to backend servers

**Integration Working:**
- Image uploads trigger AI analysis
- Search queries return relevant results
- Recommendations display with confidence scores
- All AI features respond as expected

## You're All Set!

Developed by Soumya Sourav | Ishansh | Hafsah
