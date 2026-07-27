# Adaptive Omni ML Platform

## 🚀 Complete Setup Guide

### Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn
- Git

### Backend Setup

```bash
cd /workspace/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python app/main.py
```

The backend will start on `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Frontend Setup

```bash
cd /workspace/frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

The frontend will start on `http://localhost:3000`

### Project Structure

```
/workspace
├── backend/                    # FastAPI Backend
│   ├── app/
│   │   ├── main.py            # Main API server
│   │   ├── api/               # API routes
│   │   ├── core/              # Core configuration
│   │   ├── models/            # Pydantic models
│   │   ├── schemas/           # Request/response schemas
│   │   ├── services/          # Business logic
│   │   ├── workers/           # Background tasks
│   │   └── agents/            # AI agents
│   └── requirements.txt
│
├── frontend/                   # React + TypeScript Frontend
│   ├── src/
│   │   ├── App.tsx            # Main application
│   │   ├── main.tsx           # Entry point
│   │   ├── index.css          # Global styles
│   │   ├── components/        # Reusable components
│   │   │   └── ui/            # UI components (shadcn/ui)
│   │   ├── features/          # Feature components
│   │   │   └── dashboard/     # Dashboard feature
│   │   ├── layouts/           # Layout components
│   │   ├── pages/             # Page components
│   │   ├── stores/            # Zustand stores
│   │   └── lib/               # Utilities
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── index.html
│
├── src/adaptive_ml/            # Core ML library
│   ├── qwen_omni/             # Qwen2.5-Omni integration
│   ├── training/              # Training utilities
│   ├── evaluation/            # Evaluation metrics
│   ├── memory/                # Replay buffers
│   └── models/                # Model adapters
│
├── ml/                         # ML model directories
├── data/                       # Data storage
├── models/                     # Saved models
├── experiments/                # Experiment tracking
├── configs/                    # Configuration files
└── tests/                      # Test suite
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/api/health` | System health metrics |
| GET | `/api/dashboard` | Complete dashboard data |
| GET | `/api/data-sources` | List data sources |
| POST | `/api/data-sources` | Create data source |
| GET | `/api/datasets` | List datasets |
| GET | `/api/experiments` | List experiments |
| POST | `/api/experiments` | Create experiment |
| GET | `/api/models` | List model versions |
| GET | `/api/models/current` | Get current model |
| POST | `/api/training/start` | Start training |
| POST | `/api/training/stop` | Stop training |
| GET | `/api/learning/gaps` | Detect knowledge gaps |
| POST | `/api/models/promote` | Promote model |
| POST | `/api/models/rollback` | Rollback model |

### Features Implemented

#### Frontend (v1 - Foundation)
- ✅ Modern React + TypeScript setup
- ✅ Vite build system
- ✅ Tailwind CSS styling
- ✅ shadcn/ui components
- ✅ Zustand state management
- ✅ Responsive layout with sidebar navigation
- ✅ Dashboard page with live metrics
- ✅ Placeholder pages for all major sections
- ✅ Real-time system health monitoring display

#### Backend (v1 - Foundation)
- ✅ FastAPI server with CORS
- ✅ RESTful API endpoints
- ✅ Pydantic data models
- ✅ Mock data for development
- ✅ Health check endpoint
- ✅ Dashboard data endpoint
- ✅ Data source management
- ✅ Dataset management
- ✅ Experiment tracking
- ✅ Model registry
- ✅ Training control endpoints
- ✅ Knowledge gap detection endpoint

#### Core ML Library (Existing)
- ✅ Qwen2.5-Omni-3B integration foundation
- ✅ Continual learning algorithms (EWC, MAS, SI)
- ✅ Experience replay buffers
- ✅ Knowledge distillation
- ✅ Dynamic LoRA adapters
- ✅ Model registry and versioning
- ✅ Evaluation metrics
- ✅ Drift detection
- ✅ Promotion controller

### Next Steps

1. **Connect Frontend to Backend**
   - Replace mock data with API calls
   - Implement real-time updates via WebSocket

2. **Implement Data Acquisition Engine**
   - Web scrapers
   - YouTube transcript extractor
   - PDF/document parser
   - GitHub connector
   - Custom connector SDK

3. **Build Learning Centre**
   - Training job management
   - Continual learning configuration
   - Anti-forgetting controls
   - Progress visualization

4. **Complete Testing Lab**
   - Benchmark suites
   - Regression testing
   - Model comparison
   - A/B testing

5. **Enhance Model Registry**
   - Model cards
   - Version comparison
   - Export/import functionality
   - Deployment management

6. **Add Multi-Agent System**
   - Research agent
   - Data agent
   - Verification agent
   - Evaluation agent
   - Supervisor agent

7. **Implement Knowledge Graph**
   - Entity extraction
   - Relationship mapping
   - Temporal knowledge tracking
   - Contradiction detection

### Development Commands

#### Backend
```bash
# Run server
python backend/app/main.py

# Run tests
pytest tests/

# Check code style
flake8 backend/
```

#### Frontend
```bash
# Development server
npm run dev

# Build for production
npm run build

# Lint code
npm run lint
```

### Architecture Principles

1. **Never silently replace production model** - All model changes require approval
2. **Complete audit trail** - Every action is logged and traceable
3. **Anti-forgetting protection** - Multiple layers of catastrophic forgetting prevention
4. **Data quality gates** - All data passes through quality validation
5. **Human-in-the-loop** - Configurable approval levels for autonomous operations
6. **Model versioning** - Complete lineage tracking for every model version

---

**Tagline**: An AI That Learns Continuously Without Forgetting What It Already Knows.
