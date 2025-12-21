# ProStruct: Stamp Extractor

A semi functional web application that automatically detects and extracts engineer approval stamps from PDF structural drawings. The tool uses computer vision and OCR to identify circular/concentric circle stamps, extract engineer names and license numbers, and display results with visual overlays.

## 🎯 Features

- **PDF Upload & Page Selection**: Upload PDF drawings and navigate through pages
- **Automatic Stamp Detection**: Uses layout heuristics to detect approval/engineer stamps in the title block
- **Smart Region Detection**: Focuses OCR on the right 40% of the page (top 70% of that region) where stamps typically appear
- **Circle Detection**: Detects circular and concentric circle patterns characteristic of engineer stamps
- **OCR Extraction**: Performs OCR on detected regions to extract:
  - Engineer names
  - License numbers
- **Visual Overlays**: Displays bounding boxes on detected stamps and search regions
- **Structured JSON Output**: Returns clean, structured data including coordinates, names, and license numbers
- **Multiple Stamp Support**: Handles multiple stamps per page

## 🏗️ Project Structure

```
ProStruct/
├── backend/                 # FastAPI backend server
│   ├── main.py             # Main API server and detection logic
│   ├── requirements.txt    # Python dependencies
│   └── temp_uploads/      # Temporary PDF storage
├── frontend/               # React + Vite frontend
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   ├── api.js         # API client functions
│   │   └── index.css      # Styles
│   ├── package.json       # Node.js dependencies
│   └── vite.config.js     # Vite configuration
└── README.md              # This file
```

## 🛠️ Technologies Used

### Backend
- **FastAPI**: Modern Python web framework for building APIs
- **PyMuPDF (fitz)**: PDF processing and rendering
- **OpenCV**: Computer vision for image processing and circle detection
- **OCR.space API**: Cloud-based OCR service for text extraction
- **NumPy**: Numerical computations
- **Uvicorn**: ASGI server

### Frontend
- **React 19**: UI library
- **Vite**: Build tool and dev server
- **Axios**: HTTP client
- **Tailwind CSS**: Utility-first CSS framework

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

1. **Python 3.8+** ([Download](https://www.python.org/downloads/))
2. **Node.js 16+** and npm ([Download](https://nodejs.org/))
3. **OCR.space API Key** (optional but recommended):
   - Get a free API key from [OCR.space](https://ocr.space/ocrapi/freekey)
   - Free tier includes 25,000 requests per day
   - The default key "helloworld" works but has rate limits

## 🚀 Local Setup Instructions

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd ProStruct
```

### Step 2: Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure OCR.space API Key** (recommended):
   
   Create a `.env` file in the `backend` directory:
   ```env
   OCR_SPACE_API_KEY=your_api_key_here
   ```
   
   You can get a free API key from [OCR.space](https://ocr.space/ocrapi/freekey). 
   If not set, the default key "helloworld" will be used (has rate limits).

5. **Start the backend server**:
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The backend will start at `http://localhost:8000`

### Step 3: Frontend Setup

1. **Open a new terminal** and navigate to frontend directory:
   ```bash
   cd frontend
   ```

2. **Install Node.js dependencies**:
   ```bash
   npm install
   ```

3. **Configure API URL** (if backend is not on localhost:8000):
   
   Create a `.env` file in the `frontend` directory:
   ```env
   VITE_API_URL=http://localhost:8000
   ```

4. **Start the development server**:
   ```bash
   npm run dev
   ```

   The frontend will start at `http://localhost:5173` (or another port if 5173 is busy)

### Step 4: Access the Application

Open your browser and navigate to:
```
http://localhost:5173
```

## 📖 Usage Guide

### Using the Web Interface

1. **Upload PDF**: Click the file input and select a PDF drawing file
2. **Select Page**: Use the page selector or navigation buttons to choose a page
3. **Detect Stamps**: Click the "Detect Stamp" button
4. **View Results**: 
   - See bounding boxes overlaid on detected stamps
   - View extracted engineer names and license numbers
   - Check the JSON output below

### Understanding the Output

The application returns structured JSON with the following format:

**Single Stamp:**
```json
{
  "page": 1,
  "symbol_type": "approval_stamp",
  "bounding_box": [x, y, width, height],
  "engineer_name": "THOMAS J. MAHANNA",
  "license_number": "39479",
  "units": "pixels"
}
```

**Multiple Stamps:**
```json
[
  {
    "page": 1,
    "symbol_type": "approval_stamp",
    "bounding_box": [x1, y1, w1, h1],
    "engineer_name": "Engineer Name 1",
    "license_number": "12345",
    "units": "pixels"
  },
  {
    "page": 1,
    "symbol_type": "approval_stamp",
    "bounding_box": [x2, y2, w2, h2],
    "engineer_name": "Engineer Name 2",
    "license_number": "67890",
    "units": "pixels"
  }
]
```

### Terminal Output

When processing pages, the backend prints formatted JSON output to the terminal:

```
================================================================================
JSON OUTPUT:
================================================================================
{
  "page": 1,
  "symbol_type": "approval_stamp",
  "bounding_box": [1234, 567, 200, 200],
  "engineer_name": "THOMAS J. MAHANNA",
  "license_number": "39479",
  "units": "pixels"
}
================================================================================
```

## 🔌 API Endpoints

### `POST /upload`
Upload a PDF file and get file metadata.

**Request**: Multipart form data with PDF file
**Response**:
```json
{
  "file_id": "uuid-string",
  "page_count": 5,
  "filename": "drawing.pdf",
  "pages": [
    {"index": 0, "width": 792, "height": 612},
    ...
  ]
}
```

### `GET /page/{file_id}/{page_index}`
Get a high-resolution image of a specific page.

**Response**: PNG image (image/png)

### `POST /process`
Detect stamps and extract engineer information from a page.

**Request Body**:
```json
{
  "filename": "file-id-uuid",
  "page_index": 0
}
```

**Response**: See JSON output format above

### `GET /crop/{file_id}/{page_index}?x={x}&y={y}&w={w}&h={h}`
Get a cropped region of a page.

**Query Parameters**:
- `x`, `y`: Top-left coordinates
- `w`, `h`: Width and height

**Response**: PNG image of cropped region

## 🔍 How Detection Works

1. **Region Focus**: The system focuses on the right 40% of the page (where title blocks typically are)
2. **Vertical Limiting**: Within that region, it searches the top 70% vertically
3. **Circle Detection**: Uses Hough Circle Transform to detect circular patterns
4. **Contour Analysis**: Analyzes contours for circularity and complexity
5. **OCR Processing**: Performs OCR on detected regions with multiple preprocessing methods
6. **Text Extraction**: Extracts engineer names and license numbers using pattern matching

## 🐛 Troubleshooting

### Backend Issues

**OCR.space API errors**:
- Check your API key in `.env` file
- Free tier has rate limits (25,000 requests/day)
- Ensure you have internet connection (API requires network access)
- If rate limited, wait or upgrade to a paid plan

**Port already in use**:
- Change the port in `main.py`: `uvicorn.run(app, host="0.0.0.0", port=8001)`

**Import errors**:
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

### Frontend Issues

**Cannot connect to backend**:
- Check that backend is running on port 8000
- Verify `VITE_API_URL` in frontend `.env` file
- Check CORS settings in backend (should allow all origins in development)

**Build errors**:
- Clear node_modules and reinstall: `rm -rf node_modules && npm install`
- Check Node.js version: `node --version` (should be 16+)

## 📝 Development Notes

### Backend Configuration

- **Render DPI**: Set to 300 DPI for high-quality OCR (configurable in `main.py`)
- **Upload Directory**: PDFs are temporarily stored in `backend/temp_uploads/`
- **CORS**: Currently allows all origins (update for production)

### Frontend Configuration

- **API URL**: Configured via environment variable `VITE_API_URL`
- **Build**: Run `npm run build` to create production build
- **Preview**: Run `npm run preview` to preview production build


